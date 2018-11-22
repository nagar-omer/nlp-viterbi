from scipy.sparse import csr_matrix
from submit_MEMM.ExtractFeatures import SuffixPrefix, EmmisionFtr, TransitionFtr, CombinationsWordsPos
START = "START"


class DataLoader:
    def __init__(self, src_file, feature_map, ftr_builders):
        self._ftr_builders = ftr_builders
        self._labels_to_idx = {}
        self._labels = []
        self._features = []
        self._features_to_idx = {}
        self._load_map_file(feature_map)
        self._data = self._file_to_array(src_file)

    def _load_map_file(self, map_file):
        map_file = open(map_file, "rt")
        for row in map_file:
            key, val = row.split()
            if "LABEL_" in key:
                self._labels_to_idx[key.replace("LABEL_", "")] = int(val)
            else:
                self._features_to_idx[key] = int(val)
        self._labels = [label for label, idx in sorted(self._labels_to_idx.items(), key=lambda x: x[1])]
        self._features = [ftr for ftr, idx in sorted(self._features_to_idx.items(), key=lambda x: x[1])]

    def __len__(self):
        return len(self._data)

    @property
    def label_list(self):
        return self._labels

    @property
    def data(self):
        for all_words, all_pos in self._data:
            yield all_pos, all_words

    def _to_str(self, all_words, prev_pos, i):
        ftr_str = ""
        for builder in self._ftr_builders:
            ftr_str += builder.ftr_str(all_words, prev_pos, i)
        return ftr_str

    def to_sparse(self, all_words, prev_pos, i):
        sparse_str = self._to_str(all_words, prev_pos, i).split()
        example = {ftr_name: val for ftr_name, val in [x.split("=", 1) for x in sparse_str]}
        sparse_col = []
        for ftr, val in sorted(example.items(), key=(lambda x: self._features_to_idx[x[0] + "=" + x[1]]
                               if x[0] + "=" + x[1] in self._features_to_idx else 0)):
            ftr_name = ftr + "=" + val
            if ftr_name in self._features_to_idx:
                sparse_col.append(self._features_to_idx[ftr_name])
        data = [1 for _ in range(len(sparse_col))]
        sparse_row = [0 for _ in range(len(sparse_col))]
        return csr_matrix((data, (sparse_row, sparse_col)), shape=(1, self.num_features))

    @property
    def num_features(self):
        return len(self._features)

    @property
    def num_labels(self):
        return len(self._labels)

    def idx_to_label(self, idx):
        return self._labels[idx]

    def label_to_idx(self, label):
        return self._labels_to_idx[label]

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
            data.append((all_words, all_pos))
        return data


if __name__ == "__main__":
    builders = [TransitionFtr(), EmmisionFtr(), SuffixPrefix(), CombinationsWordsPos()]
    loader = DataLoader("data/ass1-tagger-train", "data/ass1-tagger-test", builders)
    e = 1
