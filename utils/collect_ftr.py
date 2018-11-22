from utils.ftr_builders import CostumeFtr, CombinationsWordsPos, SuffixPrefix, EmmisionFtr, TransitionFtr
START = "START"


class FtrCollector:
    def __init__(self):
        self._features = []
        self._features_to_idx = {}
        self._labels_to_idx = {}
        self._labels = []

    def _read_ftr_file(self, src_file, pick_labels=False):
        all_labels = []
        all_examples = []
        src = open(src_file, "rt")
        for row in src:
            # read feature for single example
            row = row.split()
            label = row[0]
            all_labels.append(label)
            ftr_dict = {ftr_name: val for ftr_name, val in [x.split("=", 1) for x in row[1:]]}
            all_examples.append((label, ftr_dict))
        # update labels
        self._labels = list(sorted(set(all_labels))) if pick_labels else self._labels
        self._labels_to_idx = {label: i for i, label in enumerate(self._labels)} if pick_labels else self._labels_to_idx
        return all_examples

    def _train_builders(self, train_file, builders):
        # open features file
        data = self._read_ftr_file(train_file, pick_labels=True)
        for label, example in data:
            for builder in builders:
                # learn examples - (not ml just picking best representable features)
                builder.learn_example(label, example)
        # after viewing train.. get chosen features
        ftr_list = []
        for builder in builders:
            ftr_list += [str(ftr_name) for ftr_name in builder.get_ftrs()]
        self._features = list(sorted(ftr_list))
        self._features_to_idx = {ftr: i for i, ftr in enumerate(ftr_list)}

    def load_map_file(self, map_file):
        map_file = open(map_file, "rt")
        for row in map_file:
            key, val = row.split()
            if "LABEL_" in key:
                self._labels_to_idx[key.repace("LABEL_", "")] = int(val)
            else:
                self._features_to_idx[key] = int(val)
        self._labels = [label for label, idx in sorted(self._labels_to_idx.items(), key=lambda x: x[1])]
        self._features = [ftr for ftr, idx in sorted(self._features_to_idx.items(), key=lambda x: x[1])]

    def pick_ftr_by_train(self, src_train, builders):
        self._train_builders(src_train, builders)

    def create_map_file(self, out_name="map_file"):
        out_file = open(out_name, "wt")
        for key, val in self._features_to_idx.items():
            out_file.write(str(key) + " " + str(val) + "\n")
        for label, idx in self._labels_to_idx.items():
            out_file.write("LABEL_" + str(label) + " " + str(idx) + "\n")

    def create_sparse_vec_file(self, src_file, out_name="features_sparse"):
        out_file = open(out_name, "wt")
        data = self._read_ftr_file(src_file)

        for label, example in data:
            out_file.write(str(self._labels_to_idx[label]))
            for ftr, val in sorted(example.items(), key=(lambda x: self._features_to_idx[x[0] + "=" + x[1]]
                                                         if x[0] + "=" + x[1] in self._features_to_idx else 0)):
                ftr_name = ftr + "=" + val
                if ftr_name in self._features_to_idx:
                    out_file.write(" " + str(self._features_to_idx[ftr_name]) + ":1")
            out_file.write("\n")


if __name__ == "__main__":
    fc = FtrCollector()
    builders = [TransitionFtr(), EmmisionFtr(), SuffixPrefix(), CombinationsWordsPos(), CostumeFtr()]
    fc.pick_ftr_by_train("ouuut.txt", builders)
    fc.create_map_file()
    fc.create_sparse_vec_file("ouuut.txt")
    e = 1
    # loader = FtrCollector("data/ass1-tagger-train", "data/ass1-tagger-test", builders)