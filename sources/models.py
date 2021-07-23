import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random

import config
import utils


def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    """
    initialize the weight and bias(if) of the given linear layer
    :param linear: linear layer
    :return:
    """
    linear.weight.data.normal_(std=config.init_normal_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.init_normal_std)


def init_wt_normal(wt):
    """
    initialize the given weight following the normal distribution
    :param wt: weight to be normal initialized
    :return:
    """
    wt.data.normal_(std=config.init_normal_std)


def init_wt_uniform(wt):
    """
    initialize the given weight following the uniform distribution
    :param wt: weight to be uniform initialized
    :return:
    """
    wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)


class Encoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        embedded = self.embedding(inputs)  # [T, B, embedding_dim]
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class ReduceHidden(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(ReduceHidden, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        init_linear_wt(self.linear)

    def forward(self, code_hidden, ast_hidden):
        """

        :param code_hidden: hidden state of code encoder, [1, B, H]
        :param ast_hidden: hidden state of ast encoder, [1, B, H]
        :return: [1, B, H]
        """
        hidden = torch.cat((code_hidden, ast_hidden), dim=2)
        hidden = self.linear(hidden)
        hidden = F.relu(hidden)
        return hidden


class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)  # [H]
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        forward the net
        :param hidden: the last hidden state of encoder, [1, B, H]
        :param encoder_outputs: [T, B, H]
        :return: softmax scores, [B, 1, T]
        """
        time_step, batch_size, _ = encoder_outputs.size()
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)  # [B, T, H]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B, T, H]

        attn_energies = self.score(h, encoder_outputs)  # [B, T]
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)  # [B, 1, T]

        return attn_weights

    def score(self, hidden, encoder_outputs):
        """
        calculate the attention scores of each word
        :param hidden: [B, T, H]
        :param encoder_outputs: [B, T, H]
        :return: energy: scores of each word in a batch, [B, T]
        """
        # after cat: [B, T, 2/3*H]
        # after attn: [B, T, H]
        # energy: [B, T, H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # [B, T, H]
        energy = energy.transpose(1, 2)  # [B, H, T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B, 1, H]
        energy = torch.bmm(v, energy)  # [B, 1, T]
        return energy.squeeze(1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=config.hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        self.source_attention = Attention()
        self.code_attention = Attention()
        self.ast_attention = Attention()
        self.gru = nn.GRU(config.embedding_dim + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2 * self.hidden_size, config.nl_vocab_size)

        if config.use_pointer_gen:
            self.p_gen_linear = nn.Linear(2 * self.hidden_size + config.embedding_dim, 1)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        init_linear_wt(self.out)

    def forward(self, inputs, last_hidden, source_outputs, code_outputs, ast_outputs,
                extend_source_batch, extra_zeros):
        """
        forward the net
        :param inputs: word input of current time step, [B]
        :param last_hidden: last decoder hidden state, [1, B, H]
        :param source_outputs: outputs of source encoder, [T, B, H]
        :param code_outputs: outputs of code encoder, [T, B, H]
        :param ast_outputs: outputs of ast encoder, [T, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: output: [B, nl_vocab_size]
                hidden: [1, B, H]
                attn_weights: [B, 1, T]
        """
        embedded = self.embedding(inputs).unsqueeze(0)  # [1, B, embedding_dim]
        # embedded = self.dropout(embedded)

        # get attn weights of source
        # calculate and add source context in order to update attn weights during training
        source_attn_weights = self.source_attention(last_hidden, source_outputs)  # [B, 1, T]
        source_context = source_attn_weights.bmm(source_outputs.transpose(0, 1))  # [B, 1, H]
        source_context = source_context.transpose(0, 1)  # [1, B, H]

        code_attn_weights = self.code_attention(last_hidden, code_outputs)  # [B, 1, T]
        code_context = code_attn_weights.bmm(code_outputs.transpose(0, 1))  # [B, 1, H]
        code_context = code_context.transpose(0, 1)  # [1, B, H]

        ast_attn_weights = self.ast_attention(last_hidden, ast_outputs)  # [B, 1, T]
        ast_context = ast_attn_weights.bmm(ast_outputs.transpose(0, 1))  # [B, 1, H]
        ast_context = ast_context.transpose(0, 1)  # [1, B, H]

        # make ratio between source code and construct is 1: 1
        context = 0.5 * source_context + 0.5 * code_context + ast_context  # [1, B, H]

        p_gen = None
        if config.use_pointer_gen:
            # calculate p_gen
            p_gen_input = torch.cat([context, last_hidden, embedded], dim=2)  # [1, B, 2*H+E]
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)  # [1, B, 1]
            p_gen = p_gen.squeeze(0)  # [B, 1]

        rnn_input = torch.cat([embedded, context], dim=2)  # [1, B, embedding_dim + H]
        outputs, hidden = self.gru(rnn_input, last_hidden)  # [1, B, H] for both

        outputs = outputs.squeeze(0)  # [B, H]
        context = context.squeeze(0)  # [B, H]

        vocab_dist = self.out(torch.cat([outputs, context], 1))  # [B, nl_vocab_size]
        vocab_dist = F.softmax(vocab_dist, dim=1)  # P_vocab, [B, nl_vocab_size]

        if config.use_pointer_gen:
            vocab_dist_ = p_gen * vocab_dist  # [B, V]
            source_attn_weights_ = source_attn_weights.squeeze(1)  # [B, T]
            attn_dist = (1 - p_gen) * source_attn_weights_  # [B, T]

            # if has extra words
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], dim=1)  # [B, V+max_oov_num]

            # vocab_dist[i][extend_source_batch[i][j]] += attn_dist[i][j]
            # for single batch:
            # vocab_dist[extend_source_batch[j]] += attn_dist[j]
            # vocab_dist: [B, V+max_oov_num]
            # extend_source_batch: [B, T]
            # attn_dist: [B, T]
            final_dist = vocab_dist_.scatter_add(1, extend_source_batch, attn_dist)

        else:
            final_dist = vocab_dist

        final_dist = torch.log(final_dist + config.eps)

        return final_dist, hidden, source_attn_weights, code_attn_weights, ast_attn_weights, p_gen


class Model(nn.Module):

    def __init__(self, source_vocab_size, code_vocab_size, ast_vocab_size, nl_vocab_size, model=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        self.source_vocab_size = source_vocab_size
        self.code_vocab_size = code_vocab_size
        self.ast_vocab_size = ast_vocab_size
        self.is_eval = is_eval

        # init models
        self.source_encoder = Encoder(self.source_vocab_size)
        self.code_encoder = Encoder(self.code_vocab_size)
        self.ast_encoder = Encoder(self.ast_vocab_size)
        self.reduce_hidden = ReduceHidden()
        self.decoder = Decoder(nl_vocab_size)

        if model:
            assert isinstance(model, str) or isinstance(model, dict)
            if isinstance(model, str):
                model = torch.load(model)
            self.load_state_dict(model)

        if config.use_cuda:
            self.cuda()

        if is_eval:
            self.eval()

    def forward(self, batch, batch_size, nl_vocab, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        # batch: [T, B]
        source_batch, source_seq_lens, code_batch, code_seq_lens, \
            ast_batch, ast_seq_lens, nl_batch, nl_seq_lens = batch.get_regular_input()

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        source_outputs, source_hidden = self.source_encoder(source_batch, source_seq_lens)
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)
        ast_outputs, ast_hidden = self.ast_encoder(ast_batch, ast_seq_lens)

        # data for decoder
        # source_hidden = source_hidden[:1]
        code_hidden = code_hidden[0] + code_hidden[1]   # [B, H]
        code_hidden = code_hidden.unsqueeze(0)          # [1, B, H]
        ast_hidden = ast_hidden[0] + ast_hidden[1]  # [B, H]
        ast_hidden = ast_hidden.unsqueeze(0)       # [1, B, H]
        decoder_hidden = self.reduce_hidden(code_hidden, ast_hidden)  # [1, B, H]

        if is_test:
            return source_outputs, code_outputs, ast_outputs, decoder_hidden

        if nl_seq_lens is None:
            max_decode_step = config.max_decode_steps
        else:
            max_decode_step = max(nl_seq_lens)

        decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=nl_vocab)  # [B]

        extend_source_batch = None
        extra_zeros = None
        if config.use_pointer_gen:
            extend_source_batch, _, extra_zeros = batch.get_pointer_gen_input()
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size + batch.max_oov_num),
                                          device=config.device)
        else:
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size), device=config.device)

        for step in range(max_decode_step):
            # decoder_outputs: [B, nl_vocab_size]
            # decoder_hidden: [1, B, H]
            # attn_weights: [B, 1, T]
            decoder_output, decoder_hidden, source_attn_weights, code_attn_weights, ast_attn_weights, _ = self.decoder(
                inputs=decoder_inputs,
                last_hidden=decoder_hidden,
                source_outputs=source_outputs,
                code_outputs=code_outputs,
                ast_outputs=ast_outputs,
                extend_source_batch=extend_source_batch,
                extra_zeros=extra_zeros)
            decoder_outputs[step] = decoder_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
                # use teacher forcing, ground truth to be the next input
                decoder_inputs = nl_batch[step]
            else:
                # output of last step to be the next input
                _, indices = decoder_output.topk(1)  # [B, 1]
                if config.use_pointer_gen:
                    word_indices = indices.squeeze(1).detach().cpu().numpy()  # [B]
                    decoder_inputs = []
                    for index in word_indices:
                        decoder_inputs.append(utils.tune_up_decoder_input(index, nl_vocab))
                    decoder_inputs = torch.tensor(decoder_inputs, device=config.device)
                else:
                    decoder_inputs = indices.squeeze(1).detach()  # [B]
                    decoder_inputs = decoder_inputs.to(config.device)

        return decoder_outputs
