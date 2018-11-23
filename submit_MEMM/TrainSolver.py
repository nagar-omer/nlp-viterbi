import sys
import os
sys.path.insert(0, os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], ".."))
from utils.log_linear_model import LogLinear
SIZE_FTR = 5000


def create_model_file(saprse_src, out_name):
    LogLinear(train_src=saprse_src).create_model_file(out_name=out_name)


if __name__ == "__main__":
    # create_partial_file("features_sparse")
    # create_partial_file("ouuut.txt")
    args = sys.argv
    if len(args) < 2:
        print("input\t\tTrainSolver feature_vecs_file model_file")
    create_model_file(args[0], args[1])
