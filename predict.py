from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

import os
import numpy as np

import utils
import config
import models


class PredictDataset(Dataset):

    def __init__(self):
        # get lines
        with open('input/predict.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.source = lines[0].lower().strip().split(' ')
            self.code = lines[1].lower().strip().split(' ')
            self.ast = lines[2].lower().strip().split(' ')

        self.sources = [self.source]
        self.codes = [self.code]
        self.asts = [self.ast]
        self.nls = [['None']]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sources[index], self.codes[index], self.asts[index], self.nls[index]

    def get_dataset(self):
        return self.sources, self.codes, self.asts, self.nls

    def get_source(self):
        return self.source + ['</s>']


class BeamNode(object):

    def __init__(self, sentence_indices, log_probs, hidden, coverage, attn_weights, p_gens):
        """

        :param sentence_indices: indices of words of current sentence (from root to current node)
        :param log_probs: log prob of node of sentence
        :param hidden: [1, 1, H]
        """
        self.sentence_indices = sentence_indices
        self.log_probs = log_probs
        self.hidden = hidden
        self.coverage = coverage
        self.attn_weights = attn_weights if attn_weights else []
        self.p_gens = p_gens if p_gens else []

    def extend_node(self, word_index, log_prob, hidden, coverage, attn_weights, p_gen):
        return BeamNode(sentence_indices=self.sentence_indices + [word_index],
                        log_probs=self.log_probs + [log_prob],
                        hidden=hidden,
                        coverage=coverage,
                        attn_weights=self.attn_weights + [attn_weights],
                        p_gens=self.p_gens + [p_gen])

    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.sentence_indices)

    def word_index(self):
        return self.sentence_indices[-1]


class Predict(object):

    def __init__(self, model):

        # vocabulary
        self.source_vocab = utils.load_vocab_pk(config.source_vocab_path)
        self.source_vocab_size = len(self.source_vocab)
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)

        self.dataset = PredictDataset()
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=1,
                                     collate_fn=lambda *args: utils.collate_fn(args,
                                                                               source_vocab=self.source_vocab,
                                                                               code_vocab=self.code_vocab,
                                                                               ast_vocab=self.ast_vocab,
                                                                               nl_vocab=self.nl_vocab))

        # model
        if isinstance(model, str):
            self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                      code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                      code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Test\' must be file name or state_dict of the model.')

    def run_predict(self):
        self.predict()

    def predict(self):
        for index_batch, batch in enumerate(self.dataloader):

            with torch.no_grad():
                # outputs: [T, B, H]
                # hidden: [1, B, H]
                source_outputs, code_outputs, ast_outputs, decoder_hidden = \
                    self.model(batch, 1, self.nl_vocab, is_test=True)

                extend_source_batch = None
                extra_zeros = None
                if config.use_pointer_gen:
                    extend_source_batch, _, extra_zeros = batch.get_pointer_gen_input()

                # decode
                batch_sentences, attn_weights, p_gens = self.beam_decode(batch_size=1,
                                                                         source_outputs=source_outputs,
                                                                         code_outputs=code_outputs,
                                                                         ast_outputs=ast_outputs,
                                                                         decoder_hidden=decoder_hidden,
                                                                         extend_source_batch=extend_source_batch,
                                                                         extra_zeros=extra_zeros)

                # translate indices into words both for candidates
                candidates = self.translate_indices(batch_sentences, batch.batch_oovs)

                print(candidates)
                # attn_weights: [len(candidate), len(source)]
                self.plot_attention(attn_weights, candidates[0], p_gens)

    def beam_decode(self, batch_size, source_outputs: torch.Tensor, code_outputs: torch.Tensor,
                    ast_outputs: torch.Tensor, decoder_hidden: torch.Tensor, extend_source_batch, extra_zeros):
        """
        beam decode for one batch, feed one batch for decoder
        :param batch_size:
        :param source_outputs: [T, B, H]
        :param code_outputs: [T, B, H]
        :param ast_outputs: [T, B, H]
        :param decoder_hidden: [1, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: batch_sentences, [B, config.beam_top_sentence]
        """
        batch_sentences = []

        for index_batch in range(batch_size):
            # for each input sentence
            single_decoder_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)  # [1, 1, H]
            single_source_output = source_outputs[:, index_batch, :].unsqueeze(1)   # [T, 1, H]
            single_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            single_ast_output = ast_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]

            single_extend_source = extend_source_batch[index_batch]     # [T]
            single_extra_zeros = extra_zeros[index_batch]   # [max_oov_num]

            single_coverage = None
            if config.use_coverage:
                single_coverage = torch.zeros((1, config.max_code_length), device=config.device)   # [1, T]

            root = BeamNode(sentence_indices=[utils.get_sos_index(self.nl_vocab)],
                            log_probs=[0.0],
                            hidden=single_decoder_hidden,
                            coverage=single_coverage,
                            attn_weights=None,
                            p_gens=None)

            current_nodes = [root]  # list of nodes to be further extended
            final_nodes = []  # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []  # list of nodes to be extended next step

                feed_inputs = []
                feed_hidden = []
                feed_coverage = []

                # B = len(current_nodes) except eos
                extend_nodes = []
                for node in current_nodes:
                    # if current node is EOS
                    if node.word_index() == utils.get_eos_index(self.nl_vocab):
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= config.beam_width:
                            break
                        continue

                    extend_nodes.append(node)

                    decoder_input = utils.tune_up_decoder_input(node.word_index(), self.nl_vocab)

                    single_decoder_hidden = node.hidden.clone().detach()     # [1, 1, H]

                    feed_inputs.append(decoder_input)  # [B]
                    feed_hidden.append(single_decoder_hidden)   # B x [1, 1, H]

                    if config.use_coverage:
                        single_coverage = node.coverage.clone().detach()  # [1, T]
                        feed_coverage.append(single_coverage)   # [B, T]

                if len(extend_nodes) == 0:
                    break

                feed_batch_size = len(feed_inputs)
                feed_source_outputs = single_source_output.repeat(1, feed_batch_size, 1)
                feed_code_outputs = single_code_output.repeat(1, feed_batch_size, 1)
                feed_ast_outputs = single_ast_output.repeat(1, feed_batch_size, 1)

                feed_extend_source = single_extend_source.repeat(feed_batch_size, 1)
                feed_extra_zeros = single_extra_zeros.repeat(feed_batch_size, 1)

                feed_inputs = torch.tensor(feed_inputs, device=config.device)   # [B]
                feed_hidden = torch.stack(feed_hidden, dim=2).squeeze(0)    # [1, B, H]

                if config.use_coverage:
                    feed_coverage = torch.tensor(feed_coverage, device=config.device)   # [B, T]

                # decoder_outputs: [B, nl_vocab_size]
                # new_decoder_hidden: [1, B, H]
                # attn_weights: [B, 1, T]
                # coverage: [B, T]
                # p_gen: [B, 1]
                decoder_outputs, new_decoder_hidden, source_attn_weights, code_attn_weights, ast_attn_weights, \
                    next_coverage, p_gens = self.model.decoder(inputs=feed_inputs,
                                                               last_hidden=feed_hidden,
                                                               source_outputs=feed_source_outputs,
                                                               code_outputs=feed_code_outputs,
                                                               ast_outputs=feed_ast_outputs,
                                                               extend_source_batch=feed_extend_source,
                                                               extra_zeros=feed_extra_zeros,
                                                               coverage=feed_coverage)

                # get top k words
                # log_probs: [B, beam_width]
                # word_indices: [B, beam_width]
                batch_log_probs, batch_word_indices = decoder_outputs.topk(config.beam_width)

                for index_node, node in enumerate(extend_nodes):
                    log_probs = batch_log_probs[index_node]
                    word_indices = batch_word_indices[index_node]
                    hidden = new_decoder_hidden[:, index_node, :].unsqueeze(1)
                    attn_weight = source_attn_weights[index_node, :, :].squeeze(0).cpu().numpy()    # [T]
                    p_gen = p_gens[index_batch].cpu().item()   # [1]

                    coverage = None
                    if config.use_coverage:
                        coverage = next_coverage[index_node].unsqueeze(0)   # [1, T]

                    for i in range(config.beam_width):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()

                        new_node = node.extend_node(word_index=word_index,
                                                    log_prob=log_prob,
                                                    hidden=hidden,
                                                    coverage=coverage,
                                                    attn_weights=attn_weight,
                                                    p_gen=p_gen)
                        candidate_nodes.append(new_node)

                # sort candidate nodes by log_prb and select beam_width nodes
                candidate_nodes = sorted(candidate_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
                current_nodes = candidate_nodes[: config.beam_width]

            final_nodes += current_nodes
            final_nodes = sorted(final_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
            final_nodes = final_nodes[: config.beam_top_sentences]

            sentences = []
            for final_node in final_nodes:
                sentences.append(final_node.sentence_indices)
                attn_weights = final_node.attn_weights
                p_gens = final_node.p_gens

            batch_sentences.append(sentences)

        return batch_sentences, attn_weights, p_gens

    def translate_indices(self, batch_sentences, batch_oovs: list):
        """
        translate indices to words for one batch
        :param batch_sentences: [B, config.beam_top_sentences, sentence_length]
        :param batch_oovs: list of oov words list for one batch, None if not use pointer gen, [B, oov_num(variable)]
        :return:
        """
        batch_words = []
        for index_batch, sentences in enumerate(batch_sentences):
            words = []
            for indices in sentences:
                for index in indices:   # indices is a list of length 1, only loops once
                    if index not in self.nl_vocab.index2word:   # current index is out of vocabulary
                        assert batch_oovs is not None       # should not happen when not use pointer gen
                        oovs = batch_oovs[index_batch]      # oov list for current sample
                        oov_index = index - self.nl_vocab_size  # oov temp index
                        try:
                            word = oovs[oov_index]
                            config.logger.info('Pointed OOV word: {}'.format(word))
                        except IndexError:
                            # raise IndexError('Error: model produced word id', index,
                            #                  'which is corresponding to an OOV word index', oov_index,
                            #                  'but this sample only has {} OOV words.'.format(len(oovs)))
                            word = '<UNK>'
                    else:
                        word = self.nl_vocab.index2word[index]
                    if utils.is_unk(word) or not utils.is_special_symbol(word):
                        words.append(word)
            batch_words.append(words)
        return batch_words

    def plot_attention(self, attn_weights, candidate, p_gens):
        font = {'family': 'Consolas',
                'weight': 'regular',
                'size': 12}

        matplotlib.rc("font", **font)

        # attn_weights: [len(candidate), len(source)]
        source = self.dataset.get_source()
        candidate = candidate + ['</s>']

        attn_weights = np.array(attn_weights).transpose()

        df = pd.DataFrame(attn_weights, columns=candidate, index=source)

        fig1 = plt.figure(dpi=600)

        ax = fig1.add_subplot(1, 1, 1)

        cax = ax.matshow(df, interpolation='nearest', cmap='Blues')
        fig1.colorbar(cax)

        tick_spacing = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        fontdict = {'rotation': 90}  # font rotation

        ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
        ax.set_yticklabels([''] + list(df.index))

        plt.savefig('attention.png', dpi=600, bbox_inches='tight')

        # p_gens: [len(candidate)]
        p_gens = [p_gens]

        fig2 = plt.figure(dpi=600)
        df_2 = pd.DataFrame(p_gens, columns=candidate, index=['p_gen'])
        bx = fig2.add_subplot(2, 1, 2)

        cbx = bx.matshow(df_2, interpolation='nearest', cmap='Blues')
        fig2.colorbar(cbx)

        bx.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        bx.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        bx.set_xticklabels([''] + list(df_2.columns), fontdict=fontdict)
        bx.set_yticklabels([''] + list(df_2.index))

        plt.savefig('p_gen.png', dpi=600, bbox_inches = 'tight')

        plt.show()
