from torch.utils.data import Dataset

import utils
import config

import os


class CodePtrDataset(Dataset):

    def __init__(self, mode):
        assert mode in ['train', 'valid', 'test']

        # get lines
        sources = utils.load_dataset(os.path.join(config.dataset_dir, mode, config.source_name))
        codes = utils.load_dataset(os.path.join(config.dataset_dir, mode, config.code_name))
        asts = utils.load_dataset(os.path.join(config.dataset_dir, mode, config.sbt_name))
        nls = utils.load_dataset(os.path.join(config.dataset_dir, mode, config.nl_name))

        assert len(sources) == len(codes) == len(asts) == len(nls)

        self.sources, self.codes, self.asts, self.nls = utils.filter_data(sources, codes, asts, nls)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.sources[index], self.codes[index], self.asts[index], self.nls[index]

    def get_dataset(self):
        return self.sources, self.codes, self.asts, self.nls
