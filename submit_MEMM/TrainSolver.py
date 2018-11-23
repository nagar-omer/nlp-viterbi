import sys
import os
sys.path.insert(0, os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], ".."))
from utils.log_linear_model import LogLinear


def create_model_file(saprse_src, out_name):
    LogLinear(train_src=saprse_src).create_model_file(out_name=out_name)


if __name__ == "__main__":
    # create_partial_file("features_sparse")
    # create_partial_file("ouuut.txt")
    args = sys.argv
    if len(args) < 3:
        print("input\t\tTrainSolver,\t feature_vecs_file,\t model_file\n\n")
    create_model_file(args[1], args[2])
