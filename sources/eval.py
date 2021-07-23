import torch
from torch.utils.data.dataloader import DataLoader

import os
from tqdm import tqdm

import models
import data
import utils
import config
from config import logger


class BeamNode(object):

    def __init__(self, sentence_indices, log_probs, hidden):
        """

        :param sentence_indices: indices of words of current sentence (from root to current node)
        :param log_probs: log prob of node of sentence
        :param hidden: [1, 1, H]
        """
        self.sentence_indices = sentence_indices
        self.log_probs = log_probs
        self.hidden = hidden

    def extend_node(self, word_index, log_prob, hidden):
        return BeamNode(sentence_indices=self.sentence_indices + [word_index],
                        log_probs=self.log_probs + [log_prob],
                        hidden=hidden)

    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.sentence_indices)

    def word_index(self):
        return self.sentence_indices[-1]


class Test(object):

    def __init__(self, model, vocab):

        assert isinstance(model, dict) or isinstance(model, str)
        assert isinstance(vocab, tuple) or isinstance(vocab, str)

        # dataset
        logger.info('-' * 100)
        logger.info('Loading training and validation dataset')
        self.dataset = data.CodePtrDataset(mode='test')
        self.dataset_size = len(self.dataset)
        logger.info('Size of training dataset: {}'.format(self.dataset_size))

        logger.info('The dataset are successfully loaded')

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.test_batch_size,
                                     collate_fn=lambda *args: utils.collate_fn(args,
                                                                               source_vocab=self.source_vocab,
                                                                               code_vocab=self.code_vocab,
                                                                               ast_vocab=self.ast_vocab,
                                                                               nl_vocab=self.nl_vocab,
                                                                               raw_nl=True))

        # vocab
        logger.info('-' * 100)
        if isinstance(vocab, tuple):
            logger.info('Vocabularies are passed from parameters')
            assert len(vocab) == 4
            self.source_vocab, self.code_vocab, self.ast_vocab, self.nl_vocab = vocab
        else:
            logger.info('Vocabularies are read from dir: {}'.format(vocab))
            self.source_vocab = utils.load_vocab(vocab, 'source')
            self.code_vocab = utils.load_vocab(vocab, 'code')
            self.ast_vocab = utils.load_vocab(vocab, 'ast')
            self.nl_vocab = utils.load_vocab(vocab, 'nl')

        # vocabulary
        self.source_vocab_size = len(self.source_vocab)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab_size = len(self.nl_vocab)

        logger.info('Size of source vocabulary: {} -> {}'.format(self.source_vocab.origin_size, self.source_vocab_size))
        logger.info('Size of code vocabulary: {} -> {}'.format(self.code_vocab.origin_size, self.code_vocab_size))
        logger.info('Size of ast vocabulary: {}'.format(self.ast_vocab_size))
        logger.info('Size of nl vocabulary: {} -> {}'.format(self.nl_vocab.origin_size, self.nl_vocab_size))

        logger.info('Vocabularies are successfully built')

        # model
        logger.info('-' * 100)
        logger.info('Building model')
        self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                  code_vocab_size=self.code_vocab_size,
                                  ast_vocab_size=self.ast_vocab_size,
                                  nl_vocab_size=self.nl_vocab_size,
                                  is_eval=True,
                                  model=model)
        # model device
        logger.info('Model device: {}'.format(next(self.model.parameters()).device))
        # log model statistic
        logger.info('Trainable parameters: {}'.format(utils.human_format(utils.count_params(self.model))))

    def run_test(self):
        """
        start test
        :return: scores dict, key is name and value is score
        """
        logger.info('Start testing')
        scores_dict = self.test_iter()
        logger.info('Test completed')
        for name, score in scores_dict.items():
            logger.info('{}: {}'.format(name, score))

    def test_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
        total_references = []
        total_candidates = []

        out_file = None
        if config.save_test_outputs:
            try:
                out_file = open(os.path.join(config.output_root, 'test_outputs.txt'), encoding='utf-8', mode='w')
            except IOError:
                logger.error('Test details file create failed')

        with torch.no_grad():

            sample_id = 0

            p_bar = tqdm(self.dataloader, desc='[Testing...]')
            for index_batch, batch in enumerate(p_bar):

                batch_size = batch.batch_size
                references = batch.nl_batch

                # outputs: [T, B, H]
                # hidden: [1, B, H]
                source_outputs, code_outputs, ast_outputs, decoder_hidden = \
                    self.model(batch, batch_size, self.nl_vocab, is_test=True)

                extend_source_batch = None
                extra_zeros = None
                if config.use_pointer_gen:
                    extend_source_batch, _, extra_zeros = batch.get_pointer_gen_input()

                # decode
                batch_sentences = self.beam_decode(batch_size=batch_size,
                                                   source_outputs=source_outputs,
                                                   code_outputs=code_outputs,
                                                   ast_outputs=ast_outputs,
                                                   decoder_hidden=decoder_hidden,
                                                   extend_source_batch=extend_source_batch,
                                                   extra_zeros=extra_zeros)

                # translate indices into words both for candidates
                candidates = self.translate_indices(batch_sentences, batch.batch_oovs)

                total_references += references
                total_candidates += candidates

                if config.save_test_outputs:
                    for index in range(len(candidates)):
                        out_file.write('Sample {}:\n'.format(sample_id))
                        out_file.write(' '.join(['Reference:'] + references[index]) + '\n')
                        out_file.write(' '.join(['Candidate:'] + candidates[index]) + '\n')
                        out_file.write('\n')
                        sample_id += 1

            # measure
            s_blue_score, meteor_score = utils.measure(references=total_references, candidates=total_candidates)
            c_bleu = utils.corpus_bleu_score(references=total_references, candidates=total_candidates)

            avg_scores = {'c_bleu': c_bleu, 's_bleu': s_blue_score, 'meteor': meteor_score}

            if out_file:
                for name, score in avg_scores.items():
                    out_file.write(name + ': ' + str(score) + '\n')
                out_file.flush()
                out_file.close()

        return avg_scores

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
            single_source_output = source_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            single_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            single_ast_output = ast_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]

            single_extend_source = extend_source_batch[index_batch]  # [T]
            single_extra_zeros = extra_zeros[index_batch]  # [max_oov_num]

            root = BeamNode(sentence_indices=[self.nl_vocab.get_sos_index()],
                            log_probs=[0.0],
                            hidden=single_decoder_hidden)

            current_nodes = [root]  # list of nodes to be further extended
            final_nodes = []  # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []  # list of nodes to be extended next step

                feed_inputs = []
                feed_hidden = []

                # B = len(current_nodes) except eos
                extend_nodes = []
                for node in current_nodes:
                    # if current node is EOS
                    if node.word_index() == self.nl_vocab.get_eos_index():
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= config.beam_width:
                            break
                        continue

                    extend_nodes.append(node)

                    decoder_input = utils.tune_up_decoder_input(node.word_index(), self.nl_vocab)

                    single_decoder_hidden = node.hidden.clone().detach()  # [1, 1, H]

                    feed_inputs.append(decoder_input)  # [B]
                    feed_hidden.append(single_decoder_hidden)  # B x [1, 1, H]

                if len(extend_nodes) == 0:
                    break

                feed_batch_size = len(feed_inputs)
                feed_source_outputs = single_source_output.repeat(1, feed_batch_size, 1)
                feed_code_outputs = single_code_output.repeat(1, feed_batch_size, 1)
                feed_ast_outputs = single_ast_output.repeat(1, feed_batch_size, 1)

                feed_extend_source = single_extend_source.repeat(feed_batch_size, 1)
                feed_extra_zeros = single_extra_zeros.repeat(feed_batch_size, 1)

                feed_inputs = torch.tensor(feed_inputs, device=config.device)  # [B]
                feed_hidden = torch.stack(feed_hidden, dim=2).squeeze(0)  # [1, B, H]

                # decoder_outputs: [B, nl_vocab_size]
                # new_decoder_hidden: [1, B, H]
                # attn_weights: [B, 1, T]
                # coverage: [B, T]
                decoder_outputs, new_decoder_hidden, source_attn_weights, code_attn_weights, ast_attn_weights, _ = \
                    self.model.decoder(inputs=feed_inputs,
                                       last_hidden=feed_hidden,
                                       source_outputs=feed_source_outputs,
                                       code_outputs=feed_code_outputs,
                                       ast_outputs=feed_ast_outputs,
                                       extend_source_batch=feed_extend_source,
                                       extra_zeros=feed_extra_zeros)

                # get top k words
                # log_probs: [B, beam_width]
                # word_indices: [B, beam_width]
                batch_log_probs, batch_word_indices = decoder_outputs.topk(config.beam_width)

                for index_node, node in enumerate(extend_nodes):
                    log_probs = batch_log_probs[index_node]
                    word_indices = batch_word_indices[index_node]
                    hidden = new_decoder_hidden[:, index_node, :].unsqueeze(1)

                    for i in range(config.beam_width):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()

                        new_node = node.extend_node(word_index=word_index,
                                                    log_prob=log_prob,
                                                    hidden=hidden)
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

            batch_sentences.append(sentences)

        return batch_sentences

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
                for index in indices:  # indices is a list of length 1, only loops once
                    if index not in self.nl_vocab.index2word:  # current index is out of vocabulary
                        assert batch_oovs is not None  # should not happen when not use pointer gen
                        oovs = batch_oovs[index_batch]  # oov list for current sample
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
