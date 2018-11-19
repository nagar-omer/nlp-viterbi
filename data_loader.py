import random
import numpy as np
from feature_builders import SuffixPrefix, EmmisionFtr, TransitionFtr, CombinationsWordsPos
from feature_extractor import FeatureExtractor
START = "START"


class DataLoader:
    def __init__(self, source_train, source_test, ftr_builders, dev_size=0.2):
        self._ftr_builders = ftr_builders
        self._features = FeatureExtractor(source_train, ftr_builders)
        self._train = self._file_to_array(source_train)
        self._test = self._file_to_array(source_test)
        self._train, self._dev = self._split_data(self._train, 1 - dev_size)

    @property
    def len_train(self):
        return len(self._train)

    @property
    def len_test(self):
        return len(self._test)

    @property
    def len_dev(self):
        return len(self._dev)

    @property
    def train(self):
        for all_words, prev_pos, i, pos in self._train:
            yield pos, self._features.to_vec(all_words, prev_pos, i, pos)

    @property
    def test(self):
        for all_words, prev_pos, i, pos in self._test:
            yield pos, self._features.to_vec(all_words, prev_pos, i, pos)

    @property
    def dev(self):
        for all_words, prev_pos, i, pos in self._dev:
            yield pos, self._features.to_vec(all_words, prev_pos, i, pos)

    @property
    def num_features(self):
        return self._features.num_features

    @property
    def num_labels(self):
        return self._features.num_labels

    def raw_data(self, data="train"):
        if data == "train":
            data = self._train
        elif data == "test":
            data = self._test
        elif data == "dev":
            data = self._dev
        for all_words, prev_pos, i, pos in data:
            yield all_words, prev_pos, i, pos

    @staticmethod
    def _file_to_array(src_file):
        data = []
        src_file = open(src_file, "rt")  # open file
        for line in src_file:
            all_words = []
            all_pos = []
            # ---------- BREAK -----------
            for w_p in line.split():  # break line to [.. (word, POS) ..]
                word, pos = w_p.rsplit("/", 1)
                all_words.append(word)
                all_pos.append(pos)
            for i, (word, pos) in enumerate(zip(all_words, all_pos)):
                data.append((all_words, all_pos[:i], i, pos))
        return data

    @staticmethod
    def _split_data(data, train_size=0.8):
        len_data = len(data)
        split = int(len_data * train_size)
        shuffled_idx = list(range(len_data))
        random.shuffle(shuffled_idx)
        train = [data[i] for i in shuffled_idx[:split]]
        test = [data[i] for i in shuffled_idx[split:]]
        return train, test

    def train_feature_conversion_file(self, out_name="ftr_conversion", data="train"):
        if data == "train":
            data = self.train
        elif data == "test":
            data = self.test
        else:
            data = self.dev
        out_file = open(out_name, "wt")
        for i, (pos, vec) in enumerate(data):
            out_file.write(str(self._features.label_to_idx(pos)))
            for arg in np.argwhere(np.array(vec) > 0):
                out_file.write(" " + str(arg[0]) + ":1")
            out_file.write("\n")


if __name__ == "__main__":
    builders = [TransitionFtr(), EmmisionFtr(), SuffixPrefix(), CombinationsWordsPos()]
    loader = DataLoader("data/ass1-tagger-train", "data/ass1-tagger-test", builders)
    loader.train_feature_conversion_file(out_name="ftr_conv_test", data="test")
    loader.train_feature_conversion_file(out_name="ftr_conv_train", data="train")
    e = 1
