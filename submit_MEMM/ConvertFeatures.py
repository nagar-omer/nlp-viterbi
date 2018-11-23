import sys
import os
sys.path.insert(0, os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], ".."))
from utils.collect_ftr import FtrCollector
from utils.ftr_builders import TransitionFtr, EmmisionFtr, SuffixPrefix, CombinationsWordsPos, CostumeFtr
from utils.log_linear_model import LogLinear
from utils.params import LEN_FTR


def create_map_and_sparse(src_file, out_sparse, out_map):
    fc = FtrCollector()
    builders = [TransitionFtr(out_dim=LEN_FTR), EmmisionFtr(out_dim=LEN_FTR), SuffixPrefix(out_dim=LEN_FTR),
                CombinationsWordsPos(out_dim=LEN_FTR), CostumeFtr()]
    fc.pick_ftr_by_train(src_file, builders)
    fc.create_map_file(out_name=out_map)
    fc.create_sparse_vec_file(src_file, out_name=out_sparse)


def create_model_file(saprse_src):
    LogLinear(train_src=saprse_src).create_model_file()


def file_len(f_name):
    with open(f_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def create_partial_file(src_file_name, top=100, bottom=100):
    src_file = open(src_file_name, "rt")
    dst_file = open(src_file_name + "_partial", "wt")
    len_src = file_len(src_file_name)
    for i, row in enumerate(src_file):
        if i < top or i >= len_src - bottom:
            dst_file.write(row)

    src_file.close()


if __name__ == "__main__":
    # create_partial_file("features_vec")
    # create_partial_file("features_map")
    args = sys.argv
    if len(args) < 4:
        print("input\t\tConvertFeatures,\t features_file,\t feature_vecs_file,\t feature_map_file\n\n")
    create_map_and_sparse(args[1], args[2], args[3])
