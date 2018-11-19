from feature_builders import FeatureBuilder, TransitionFtr, EmmisionFtr, SuffixPrefix, CombinationsWordsPos

START = "START"


class FeatureExtractor:
    def __init__(self, source_file, builders: list):
        self._source = source_file
        self._data, self._label_list = self._read_train_file()
        self._label_to_idx = {label: idx for idx, label in enumerate(self._label_list)}
        self._ftr_builders = builders
        self._ftr_list, self._ftr_to_idx = self._train_builders()

    @property
    def num_features(self):
        return len(self._ftr_list)

    @property
    def num_labels(self):
        return len(self._label_list)

    def label_to_idx(self, pos: str):
        return self._label_to_idx[pos]

    def idx_to_label(self, idx: int):
        return self._label_list[idx]

    def _read_train_file(self):
        label_list = []
        data = []
        src_file = open(self._source, "rt")  # open file
        for line in src_file:
            # ---------- BREAK -----------
            temp = []
            for w_p in line.split():  # break line to [.. (word, POS) ..]
                word, pos = w_p.rsplit("/", 1)
                temp.append((word, pos))
                label_list.append(pos)
            data.append(temp)
        src_file.close()
        return data, list(set(label_list))

    def _train_builders(self):
        for example in self._data:
            prev_pos = [START, START]
            all_words = [word for word, pos in example]
            for i, (word, pos) in enumerate(example):
                for builder in self._ftr_builders:
                    builder.learn_example(all_words, prev_pos, i, pos)
                prev_pos.append(pos)
        ftr_list = []
        for builder in self._ftr_builders:
            ftr_list += [builder.name + str(ftr_name) for ftr_name in builder.get_ftrs()]
        ftr_to_idx = {ftr: i for i, ftr in enumerate(ftr_list)}
        return ftr_list, ftr_to_idx

    def to_vec(self, all_words, prev_pos, i, pos):
        ftr_dict = {}
        for builder in self._ftr_builders:
            ftr_dict.update({builder.name + str(ftr_name): val for ftr_name, val in
                             builder.to_vec(all_words, prev_pos, i, pos).items()})
        return [ftr_dict[name] for name in sorted(self._ftr_list)]

    def to_dict(self, all_words, prev_pos, i, pos):
        ftr_dict = {}
        for builder in self._ftr_builders:
            ftr_dict.update({builder.name + str(ftr_name): val for ftr_name, val in
                             builder.to_vec(all_words, prev_pos, i, pos).items()})
        return ftr_dict

    def to_str(self, all_words, prev_pos, i, pos):
        ftr_str = ""
        for builder in self._ftr_builders:
            ftr_str += builder.ftr_str(all_words, prev_pos, i, pos)
        return ftr_str

    def features_str_file(self, out_name="ftr_out"):
        out_file = open(out_name, "wt")
        for example in self._data:
            prev_pos = [START, START]
            all_words = [word for word, pos in example]
            for i, (word, pos) in enumerate(example):
                out_file.write(pos + self.to_str(all_words, prev_pos, i, pos) + "\n")
        out_file.close()

    def feature_map_file(self, out_name="ftr_map"):
        out_file = open(out_name, "wt")
        for key, val in self._ftr_to_idx.items():
            out_file.write(str(key) + " " + str(val) + "\n")


if __name__ == "__main__":
    builders = [TransitionFtr(), EmmisionFtr(), SuffixPrefix(), CombinationsWordsPos()]
    mm = FeatureExtractor("data/ass1-tagger-train", builders)
    mm.features_str_file()
    mm.feature_map_file()
    e = 0
