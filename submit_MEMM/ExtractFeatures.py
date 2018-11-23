import os
import sys
sys.path.insert(0, os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], ".."))
from utils.ftr_builders import TransitionFtr, EmmisionFtr, SuffixPrefix, CombinationsWordsPos, CostumeFtr


FTR_SIZE = 5000
START = "START"


class FeatureExtractor:
    def __init__(self, builders):
        self._ftr_builders = builders

    def _to_str(self, all_words, prev_pos, i):
        ftr_str = ""
        for builder in self._ftr_builders:
            ftr_str += builder.ftr_str(all_words, prev_pos, i)
        return ftr_str

    @staticmethod
    def _read_file(src_file):
        data = []
        src_file = open(src_file, "rt")  # open file
        for line in src_file:
            # ---------- BREAK -----------
            temp = []
            for w_p in line.split():  # break line to [.. (word, POS) ..]
                word, pos = w_p.rsplit("/", 1)
                temp.append((word, pos))
            data.append(temp)
        src_file.close()
        return data

    def create_ftr_file_for(self, in_file, out_name="ftr_out"):
        data = self._read_file(in_file)
        out_file = open(out_name, "wt")
        for example in data:
            prev_pos = [START, START]
            all_words = [word for word, pos in example]
            for i, (word, pos) in enumerate(example):
                out_file.write(pos + self._to_str(all_words, prev_pos, i) + "\n")
                prev_pos.append(pos)
        out_file.close()


def create_ftr_file_for(src, out_name):
    builders = [TransitionFtr(out_dim=FTR_SIZE), EmmisionFtr(out_dim=FTR_SIZE), SuffixPrefix(out_dim=FTR_SIZE),
                CombinationsWordsPos(out_dim=FTR_SIZE), CostumeFtr()]
    mm = FeatureExtractor(builders)
    mm.create_ftr_file_for(src, out_name=out_name)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("input\t\tExtractFeatures corpus_file features_file")
    # create_ftr_file_for(os.path.join("..", "data", "ass1-tagger-train"), "ouuut.txt")
    create_ftr_file_for(args[0], args[1])
