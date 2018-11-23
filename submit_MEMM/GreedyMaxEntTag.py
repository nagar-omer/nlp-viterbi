import sys
import os
sys.path.insert(0, os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], ".."))
import pickle
from utils.data_loader import DataLoader
from utils.ftr_builders import TransitionFtr, CombinationsWordsPos, EmmisionFtr, CostumeFtr, SuffixPrefix

START = "START"
SIZE_FTR = 5000


def greedy_tag(to_pred, model_file, feature_map, out_name="greedy_pred"):
    out_file = open(out_name, "wt")

    model = pickle.load(open(model_file, "rb"))
    ftr_builders = [TransitionFtr(out_dim=SIZE_FTR), EmmisionFtr(out_dim=SIZE_FTR), SuffixPrefix(out_dim=SIZE_FTR),
                    CombinationsWordsPos(out_dim=SIZE_FTR), CostumeFtr()]
    dl = DataLoader(to_pred, feature_map, ftr_builders)

    all_count = 0
    true_count = 0
    len_data = len(dl)
    for j, (all_pos, all_words) in enumerate(dl.data):
        if (100 * j / len_data) % 10 == 0:
            print(str((100 * j / len_data)) + "%")
        prev_pos = [START, START]
        for i, (word, pos) in enumerate(zip(all_words, all_pos)):
            curr_pred = model.predict(dl.to_sparse(all_words, prev_pos, i))
            prev_pos.append(pos)
            all_count += 1
            curr_pred_label = dl.idx_to_label(int(curr_pred[0]))
            out_file.write(word + "/" + curr_pred_label + " ")
            true_count += 1 if pos == curr_pred_label else 0
            # print(word, pos, dl.idx_to_label(int(curr_pred[0])))
        out_file.write("\n")
    out_file.close()
    print(all_count, true_count, "\t~" + str(int(100*true_count/all_count)) + "%")


if __name__ == "__main__":
    # greedy_tag("../data/ass1-tagger-test", "model_file", "map_file")
    args = sys.argv
    if len(args) < 4:
        print("input\t\tinput_file_name, modelname, feature_map_file, out_file_name")
    greedy_tag(args[0], args[1], args[2], out_name=args[3])

