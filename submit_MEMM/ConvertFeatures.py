from utils.collect_ftr import FtrCollector
from utils.ftr_builders import TransitionFtr, EmmisionFtr, SuffixPrefix, CombinationsWordsPos, CostumeFtr
from utils.log_linear_model import LogLinear
SIZE_FTR = 5000


def create_map_and_sparse(src_file):
    fc = FtrCollector()
    builders = [TransitionFtr(out_dim=SIZE_FTR), EmmisionFtr(out_dim=SIZE_FTR), SuffixPrefix(out_dim=SIZE_FTR),
                CombinationsWordsPos(out_dim=SIZE_FTR), CostumeFtr()]
    fc.pick_ftr_by_train(src_file, builders)
    fc.create_map_file()
    fc.create_sparse_vec_file(src_file)


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
    # create_partial_file("features_sparse")
    # create_partial_file("ouuut.txt")
    create_map_and_sparse("ouuut.txt")
    create_model_file("features_sparse")
    e = 1