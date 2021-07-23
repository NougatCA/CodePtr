import time

import torch
import itertools
import os
import pickle
import numpy as np
import nltk

import config
from config import logger


class Vocab(object):

    # special vocabulary symbols
    PAD_TOKEN = '<pad>'     # padding token
    SOS_TOKEN = '<sos>'     # start of sequence
    EOS_TOKEN = '<eos>'     # end of sequence
    UNK_TOKEN = '<unk>'     # unknown token

    # default special symbols, if need additional symbols, use init parameter 'additional_special_symbols'
    START_VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self, name, additional_special_symbols=None, ignore_case=False):
        """
        Initialization Definition.
        Args:
            name (str): vocabulary name
            additional_special_symbols (list): optional, list of custom special symbols
            ignore_case (bool): optional, ignore cases if True, default False
        """
        self.ignore_case = ignore_case
        self.special_symbols = Vocab.START_VOCAB.copy()
        if additional_special_symbols:
            self.add_special_symbols(additional_special_symbols)
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.add_sentence(self.special_symbols)     # add special symbols

        self.origin_size = 0    # vocab size before trim

    def add_dataset(self, dataset):
        """
        Add a list of list of tokens.
        Args:
            dataset (list): a list object whose elements are all lists of str objects

        """
        for seq in dataset:
            for token in seq:
                self.add_word(token)

    def add_sentence(self, sentence):
        """
        Add a list of tokens.
        Args:
            sentence (list): a list object whose elements are all str objects

        """
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        """
        Add a single word.
        Args:
            word (str): str object

        """
        if self.ignore_case:
            word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, max_vocab_size):
        """
        Trim the vocabulary to the given size according to the frequency of the words.
            if the size is greater than the given size
        Args:
            max_vocab_size: max vocabulary size

        Returns:
            list:
                - words which is eliminated, list of tuples (word, count)
        """
        if self.trimmed:
            return None
        self.trimmed = True
        self.origin_size = self.num_words

        if self.num_words <= max_vocab_size:
            return None
        for special_symbol in self.special_symbols:
            self.word2count.pop(special_symbol)
        all_words = list(self.word2count.items())
        all_words = sorted(all_words, key=lambda item: item[1], reverse=True)

        keep_words = all_words[:max_vocab_size - len(self.special_symbols)]
        keep_words = self.special_symbols + [word for word, _ in keep_words]

        # trimmed words
        trimmed_words = all_words[max_vocab_size - len(self.special_symbols):]

        # reinitialize
        self.word2index.clear()
        self.word2count.clear()
        self.index2word.clear()
        self.num_words = 0
        self.add_sentence(keep_words)

        return trimmed_words

    def add_special_symbols(self, symbols: list):
        assert isinstance(symbols, list)
        for symbol in symbols:
            assert isinstance(symbol, str)
            if symbol not in self.special_symbols:
                self.special_symbols.append(symbol)

    def get_index(self, word):
        """
        Return the index of given word, if the given word is not in the vocabulary, return the index of UNK token.
        Args:
            word (str): word in str

        Returns:
            int:
                - index of the given word, UNK if OOV
        """
        if self.ignore_case:
            word = word.lower()
        return self.word2index[word] if word in self.word2index else self.word2index[Vocab.UNK_TOKEN]

    def get_pad_index(self):
        return self.word2index[Vocab.PAD_TOKEN]

    def get_sos_index(self):
        return self.word2index[Vocab.SOS_TOKEN]

    def get_eos_index(self):
        return self.word2index[Vocab.EOS_TOKEN]

    def get_unk_index(self):
        return self.word2index[Vocab.UNK_TOKEN]

    def get_word(self, index):
        """
        Return the corresponding word of the given index, if not in the vocabulary, return '<unk>'.
        Args:
            index: given index

        Returns:
            str:
                - token of the given index
        """
        return self.index2word[index] if index in self.index2word else Vocab.UNK_TOKEN

    def save(self, vocab_dir, name=None):
        path = os.path.join(vocab_dir, '{}_vocab.pk'.format(self.name) if name is None else name)
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def save_txt(self, vocab_dir, name=None):
        path = os.path.join(vocab_dir, '{}_vocab.txt'.format(self.name) if name is None else name)
        with open(path, 'w', encoding='utf-8') as file:
            for word, _ in self.word2index.items():
                file.write(word + '\n')

    def __len__(self):
        return self.num_words

    def __contains__(self, item):
        """
        Return True if the given word is in the vocab, else False.
        Args:
            item: word to query

        Returns:
            bool:
                - True if the given word is in the vocab, else False.
        """
        if self.ignore_case:
            item = item.lower()
        return item in self.word2index


class Batch(object):

    def __init__(self, source_batch, source_seq_lens, code_batch, code_seq_lens,
                 ast_batch, ast_seq_lens, nl_batch, nl_seq_lens):
        self.source_batch = source_batch
        self.source_seq_lens = source_seq_lens
        self.code_batch = code_batch
        self.code_seq_lens = code_seq_lens
        self.ast_batch = ast_batch
        self.ast_seq_lens = ast_seq_lens
        self.nl_batch = nl_batch
        self.nl_seq_lens = nl_seq_lens

        self.batch_size = len(source_seq_lens)

        # pointer gen
        self.extend_source_batch = None
        self.extend_nl_batch = None
        self.max_oov_num = None
        self.batch_oovs = None
        self.extra_zeros = None

    def get_regular_input(self):
        return self.source_batch, self.source_seq_lens, self.code_batch, self.code_seq_lens, \
               self.ast_batch, self.ast_seq_lens, self.nl_batch, self.nl_seq_lens

    def config_point_gen(self, extend_source_batch_indices, extend_nl_batch_indices, batch_oovs,
                         source_vocab, nl_vocab, raw_nl):
        self.batch_oovs = batch_oovs
        self.max_oov_num = max([len(oovs) for oovs in self.batch_oovs])

        self.extend_source_batch = pad_one_batch(extend_source_batch_indices, source_vocab)     # [T, B]
        self.extend_source_batch = self.extend_source_batch.transpose(0, 1)

        # [T, B]
        if not raw_nl:
            self.extend_nl_batch = pad_one_batch(extend_nl_batch_indices, nl_vocab)

        if self.max_oov_num > 0:
            # [B, max_oov_num]
            self.extra_zeros = torch.zeros((self.batch_size, self.max_oov_num), device=config.device)

    def get_pointer_gen_input(self):
        return self.extend_source_batch, self.extend_nl_batch, self.extra_zeros


class EarlyStopping(object):

    def __init__(self, patience, delta=0, high_record=False):
        """
        Initialize an EarlyStopping instance
        Args:
            patience: How long to wait after last time validation loss decreased
            delta: Minimum change in the monitored quantity to qualify as an improvement
            high_record: True if the improvement of the record is seen as the improvement of the performance,
                default False
        """
        self.patience = patience
        self.counter = 0
        self.record = None
        self.early_stop = False
        self.delta = delta
        self.high_record = high_record

        self.refreshed = False
        self.best_model = None
        self.best_epoch = -1

    def __call__(self, score, model, epoch):
        """
        Call this instance when get a new score
        Args:
            score (float): the new score
            model:
        """
        # first call
        if self.record is None:
            self.record = score
            self.refreshed = True
            self.best_model = model
            self.best_epoch = epoch
        # not hit the best
        elif (not self.high_record and score > self.record + self.delta) or \
                (self.high_record and score < self.record - self.delta):
            self.counter += 1
            self.refreshed = False
            logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning('Early stop')
        # hit the best
        else:
            self.record = score
            self.counter = 0
            self.refreshed = True
            self.best_model = model
            self.best_epoch = epoch


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    Computes elapsed time.
    """
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def load_vocab_pk(file_name) -> Vocab:
    """
    load pickle file by given file name
    :param file_name:
    :return:
    """
    path = os.path.join(config.vocab_root, file_name)
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, Vocab):
        raise Exception('Pickle file: \'{}\' is not an instance of class \'Vocab\''.format(path))
    return vocab


def get_timestamp():
    """
    return the current timestamp, eg. 20200222_151420
    :return: current timestamp
    """
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def load_dataset(dataset_path) -> list:
    """
    load the dataset from given path
    :param dataset_path: path of dataset
    :return: lines from the dataset
    """
    lines = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words = line.lower().strip().split(' ')
            lines.append(words)
    return lines


def filter_data(sources, codes, asts, nls):
    """
    filter the data according to the rules
    :param sources: list of tokens of source codes
    :param codes: list of tokens of split source codes
    :param asts: list of tokens of sequence asts
    :param nls: list of tokens of comments
    :return: filtered codes, asts and nls
    """
    assert len(sources) == len(codes)
    assert len(codes) == len(asts)
    assert len(asts) == len(nls)

    new_sources = []
    new_codes = []
    new_asts = []
    new_nls = []
    for i in range(len(codes)):
        source = sources[i]
        code = codes[i]
        ast = asts[i]
        nl = nls[i]

        if len(code) > config.max_code_length or len(nl) > config.max_nl_length or len(nl) < config.min_nl_length:
            continue

        new_sources.append(source)
        new_codes.append(code)
        new_asts.append(ast)
        new_nls.append(nl)
    return new_sources, new_codes, new_asts, new_nls


def init_vocab(name, lines):
    """
    initialize the vocab by given name and dataset, trim if necessary
    :param name: name of vocab
    :param lines: dataset
    :return: vocab
    """
    vocab = Vocab(name)
    for line in lines:
        vocab.add_sentence(line)
    return vocab


def init_decoder_inputs(batch_size, vocab: Vocab) -> torch.Tensor:
    """
    initialize the input of decoder
    :param batch_size:
    :param vocab:
    :return: initial decoder input, torch tensor, [batch_size]
    """
    return torch.tensor([vocab.get_sos_index()] * batch_size, device=config.device)


def filter_oov(inputs, vocab: Vocab):
    """
    replace the oov words with UNK token
    :param inputs: inputs, [time_step, batch_size]
    :param vocab: corresponding vocab
    :return: filtered inputs, numpy array, [time_step, batch_size]
    """
    for index_step, step in enumerate(inputs):
        for index_word, word in enumerate(step):
            if word >= vocab.num_words:
                inputs[index_step][index_word] = vocab.get_unk_index()
    return inputs


def get_seq_lens(batch: list) -> list:
    """
    get sequence lengths of given batch
    :param batch: [B, T]
    :return: sequence lengths
    """
    seq_lens = []
    for seq in batch:
        seq_lens.append(len(seq))
    return seq_lens


def pad_one_batch(batch: list, vocab: Vocab) -> torch.Tensor:
    """
    pad batch using _PAD token and get the sequence lengths
    :param batch: one batch, [B, T]
    :param vocab: corresponding vocab
    :return:
    """
    batch = list(itertools.zip_longest(*batch, fillvalue=vocab.get_pad_index()))
    batch = [list(b) for b in batch]
    return torch.tensor(batch, device=config.device).long()


def indices_from_batch(batch: list, vocab: Vocab) -> list:
    """
    translate the word in batch to corresponding index by given vocab, then append the EOS token to each sentence
    :param batch: batch to be translated, [B, T]
    :param vocab: Vocab
    :return: translated batch, [B, T]
    """
    indices = []
    for sentence in batch:
        indices_sentence = []
        for word in sentence:
            if word not in vocab.word2index:
                indices_sentence.append(vocab.get_unk_index())
            else:
                indices_sentence.append(vocab.word2index[word])
        indices_sentence.append(vocab.get_eos_index())
        indices.append(indices_sentence)
    return indices


def extend_indices_from_batch(source_batch: list, nl_batch: list, source_vocab: Vocab, nl_vocab: Vocab, raw_nl):
    """

    :param source_batch: [B, T]
    :param nl_batch:
    :param source_vocab:
    :param nl_vocab:
    :param raw_nl
    :return:
    """
    extend_source_batch_indices = []   # [B, T]
    extend_nl_batch_indices = []
    batch_oovs = []     # list of list of oov words in sentences
    for source, nl in zip(source_batch, nl_batch):

        oovs = []
        extend_source_indices = []
        extend_nl_indices = []
        oov_temp_index = {}     # maps the oov word to temp index

        for word in source:
            if word not in source_vocab.word2index:
                if word not in oovs:
                    oovs.append(word)
                oov_index = oovs.index(word)
                temp_index = len(source_vocab) + oov_index
                extend_source_indices.append(temp_index)
                oov_temp_index[word] = temp_index
            else:
                extend_source_indices.append(source_vocab.word2index[word])
        extend_source_indices.append(source_vocab.get_eos_index())

        if not raw_nl:
            for word in nl:
                if word not in nl_vocab.word2index:
                    if word in oov_temp_index:      # in-source oov word
                        temp_index = oov_temp_index[word]
                        extend_nl_indices.append(temp_index)
                    else:
                        extend_nl_indices.append(nl_vocab.get_unk_index())     # oov words not appear in source code
                else:
                    extend_nl_indices.append(nl_vocab.word2index[word])
            extend_nl_indices.append(nl_vocab.get_eos_index())

        extend_source_batch_indices.append(extend_source_indices)
        extend_nl_batch_indices.append(extend_nl_indices)
        batch_oovs.append(oovs)

    return extend_source_batch_indices, extend_nl_batch_indices, batch_oovs


def sort_batch(batch) -> (list, list, list):
    """
    sort one batch, return indices and sequence lengths
    :param batch: [B, T]
    :return:
    """
    seq_lens = get_seq_lens(batch)
    pos = np.argsort(seq_lens)[::-1]
    batch = [batch[index] for index in pos]
    seq_lens.sort(reverse=True)
    return batch, seq_lens, pos


def restore_encoder_outputs(outputs: torch.Tensor, pos) -> torch.Tensor:
    """
    restore the outputs or hidden of encoder by given pos
    :param outputs: [T, B, H] or [2, B, H]
    :param pos:
    :return:
    """
    rev_pos = np.argsort(pos)
    outputs = torch.index_select(outputs, 1, torch.tensor(rev_pos, device=config.device))
    return outputs


def tune_up_decoder_input(index, vocab):
    """
    replace index with unk if index is out of vocab size
    :param index:
    :param vocab:
    :return:
    """
    if index >= len(vocab):
        index = vocab.get_unk_index()
    return index


def collate_fn(batch, source_vocab, code_vocab, ast_vocab, nl_vocab, raw_nl=False):
    """
    process the batch without sorting
    :param batch: one batch, first dimension is batch, [B]
    :param source_vocab:
    :param code_vocab:
    :param ast_vocab: [B, T]
    :param nl_vocab: [B, T]
    :param raw_nl: True when test, nl_batch will not be translated and returns the raw data
    :return:
    """
    batch = batch[0]
    source_batch = []
    code_batch = []
    ast_batch = []
    nl_batch = []
    for b in batch:
        source_batch.append(b[0])
        code_batch.append(b[1])
        ast_batch.append(b[2])
        nl_batch.append(b[3])

    # transfer words to indices including oov words, and append EOS token to each sentence, list
    extend_source_batch_indices = None
    extend_nl_batch_indices = None
    batch_oovs = None
    if config.use_pointer_gen:
        # if raw_nl, extend_nl_batch_indices is a empty list
        extend_source_batch_indices, extend_nl_batch_indices, batch_oovs = extend_indices_from_batch(source_batch,
                                                                                                     nl_batch,
                                                                                                     source_vocab,
                                                                                                     nl_vocab,
                                                                                                     raw_nl)
    source_batch = indices_from_batch(source_batch, source_vocab)
    code_batch = indices_from_batch(code_batch, code_vocab)  # [B, T]
    ast_batch = indices_from_batch(ast_batch, ast_vocab)  # [B, T]
    if not raw_nl:
        nl_batch = indices_from_batch(nl_batch, nl_vocab)  # [B, T]

    source_seq_lens = get_seq_lens(source_batch)
    code_seq_lens = get_seq_lens(code_batch)
    ast_seq_lens = get_seq_lens(ast_batch)
    nl_seq_lens = get_seq_lens(nl_batch)

    # pad and transpose, [T, B], tensor
    source_batch = pad_one_batch(source_batch, source_vocab)
    code_batch = pad_one_batch(code_batch, code_vocab)
    ast_batch = pad_one_batch(ast_batch, ast_vocab)
    if not raw_nl:
        nl_batch = pad_one_batch(nl_batch, nl_vocab)

    batch = Batch(source_batch, source_seq_lens, code_batch, code_seq_lens,
                  ast_batch, ast_seq_lens, nl_batch, nl_seq_lens)

    if config.use_pointer_gen:
        batch.config_point_gen(extend_source_batch_indices,
                               extend_nl_batch_indices,
                               batch_oovs,
                               source_vocab,
                               nl_vocab,
                               raw_nl)

    return batch


def to_time(float_time):
    """
    translate float time to h, min, s and ms
    :param float_time: time in float
    :return: h, min, s, ms
    """
    time_s = int(float_time)
    time_ms = int((float_time - time_s) * 1000)
    time_h = time_s // 3600
    time_s = time_s % 3600
    time_min = time_s // 60
    time_s = time_s % 60
    return time_h, time_min, time_s, time_ms


def is_unk(word):
    if word == Vocab.UNK_TOKEN:
        return True
    return False


def is_special_symbol(word):
    if word in Vocab.START_VOCAB:
        return True
    else:
        return False


def measure(references, candidates) -> (float, float):
    """
    measures the top sentence model generated
    :param references: batch of references
    :param candidates: batch of sentences model generated
    :return: total sentence level bleu score, total meteor score
    """
    assert len(references) == len(candidates)
    batch_size = len(references)

    total_s_bleu = 0
    total_meteor = 0

    for reference, candidate in zip(references, candidates):

        # sentence level bleu score
        sentence_bleu = sentence_bleu_score(reference, candidate)
        total_s_bleu += sentence_bleu

        # meteor score
        meteor = meteor_score(reference, candidate)
        total_meteor += meteor

    return total_s_bleu / batch_size, total_meteor / batch_size


def sentence_bleu_score(reference, candidate) -> float:
    """
    calculate the sentence level bleu score, 4-gram with weights(0.25, 0.25, 0.25, 0.25)
    :param reference: tokens of reference sentence
    :param candidate: tokens of sentence generated by model
    :return: sentence level bleu score
    """
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    return nltk.translate.bleu_score.sentence_bleu(references=[reference],
                                                   hypothesis=candidate,
                                                   smoothing_function=smoothing_function.method5)


def corpus_bleu_score(references, candidates) -> float:
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    return nltk.translate.bleu_score.corpus_bleu(list_of_references=[[reference] for reference in references],
                                                 hypotheses=[candidate for candidate in candidates],
                                                 smoothing_function=smoothing_function.method5)


def meteor_score(reference, candidate):
    """
    meteor score
    :param reference:
    :param candidate:
    :return:
    """
    return nltk.translate.meteor_score.single_meteor_score(reference=' '.join(reference),
                                                           hypothesis=' '.join(candidate))


def save_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def build_word_vocab(dataset, vocab_name, ignore_case=False, max_vocab_size=None, special_symbols=None,
                     save_dir=None, save_name=None, save_txt_name=None):
    """
    Build a regular word vocab.
    Args:
        dataset (list): list of sequence of token
        vocab_name (str): name of the vocab
        ignore_case (bool): optional, True if ignore the case, default: False
        max_vocab_size (int): optional, trim the vocab to the max_vocab_size
        special_symbols (list): optional, list of str, additional special symbols except pad, sos, eos and unk
        save_dir (str): directory path to save vocab
        save_name (str): optional, save the vocab to the given name
        save_txt_name (str): optional, save the vocab as the txt file to the given path

    Returns:
        Vocab:
            - built vocab
    """
    vocab = Vocab(name=vocab_name, ignore_case=ignore_case, additional_special_symbols=special_symbols)
    vocab.add_dataset(dataset)
    if max_vocab_size:
        vocab.trim(max_vocab_size)
    if save_dir:
        vocab.save(vocab_dir=save_dir, name=save_name)
        vocab.save_txt(vocab_dir=save_dir, name=save_txt_name)
    return vocab


def load_vocab(vocab_dir, name):
    with open(os.path.join(vocab_dir, '{}_vocab.pk'.format(name)), mode='rb') as f:
        obj = pickle.load(f)
    assert isinstance(obj, Vocab)
    return obj


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def count_params(model):
    """
    Count the number of parameters of given model
    """
    return sum(p.numel() for p in model.parameters())


def time2str(float_time):
    time_h, time_min, time_s, time_ms = to_time(float_time)
    return '{}h {}min {}s {}ms'.format(time_h, time_min, time_s, time_ms)
