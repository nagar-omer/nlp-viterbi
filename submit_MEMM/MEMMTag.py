import sys
import os
sys.path.insert(0, os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], ".."))
import pickle
import numpy as np
from utils.data_loader import DataLoader
from utils.ftr_builders import TransitionFtr, EmmisionFtr, SuffixPrefix, CombinationsWordsPos, CostumeFtr
from utils.viterbi import ViterbiAlg

START = "START"
SIZE_FTR = 5000


class MEMMTagger:
    def __init__(self, to_pred, model_file, feature_map, out_name="greedy_pred"):
        self._probs = {}
        self._model = pickle.load(open(model_file, "rb"))
        ftr_builders = [TransitionFtr(out_dim=SIZE_FTR), EmmisionFtr(out_dim=SIZE_FTR), SuffixPrefix(out_dim=SIZE_FTR),
                        CombinationsWordsPos(out_dim=SIZE_FTR), CostumeFtr()]
        self._dl = DataLoader(to_pred, feature_map, ftr_builders)
        self._label_list = self._dl.label_list + [START]
        self._label_to_idx = {label: i for i, label in enumerate(self._label_list)}
        self._tagger = ViterbiAlg(self._label_list, self._prob_func)
        self._init_probs()

    # --------------------- prob_func
    # input:  word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=T/F
    # output: best_score, back pointer
    def _prob_func(self, word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=True):
        words = []
        words.append(word_sequence[curr_word_idx - 1] if curr_word_idx > 0 else START)
        words.append(word_sequence[curr_word_idx - 2] if curr_word_idx > 1 else START)
        words.append(word_sequence[curr_word_idx])
        words.append(word_sequence[curr_word_idx + 1] if curr_word_idx + 1 < len(word_sequence) else None)
        words.append(word_sequence[curr_word_idx + 2] if curr_word_idx + 2 < len(word_sequence) else None)

        if log:
            if np.std(src_prob_row) > 0:
                good_chance = np.argsort(src_prob_row)[-8:]
            else:
                good_chance = list(range(len(src_prob_row)))
        scores = {}
        for ii, prev_prev_pos in enumerate(good_chance):
            if str(([prev_prev_pos, prev_POS], words)) not in self._probs:
                sparse_vec = self._dl.to_sparse(word_sequence, [prev_prev_pos, prev_POS], curr_word_idx)
                self._probs[str(([prev_prev_pos, prev_POS], words))] = \
                    self._model.predict_proba(sparse_vec)[0]

            if ii < 3 or self._probs[str(([prev_prev_pos, prev_POS], words))][self._label_to_idx[curr_POS]] > 0.5:
                scores[prev_prev_pos] = \
                    self._probs[str(([prev_prev_pos, prev_POS], words))][self._label_to_idx[curr_POS]]

        scores = {key: src_prob_row[key] + self._my_log(val) if log else src_prob_row[key] * val for key, val
                  in scores.items()}
        back_pointer, best_score = max(scores.items(), key=lambda x: x[1])
        return best_score, back_pointer

    def _init_probs(self):
        print("loadig model...")
        for j, (all_pos, all_words) in enumerate(self._dl.data):

            len_data = len(self._dl)
            if (100 * j / len_data) % 10 == 0:
                print(str((100 * j / len_data)) + "%")
            prev_pos = [START, START]
            for i, (word, pos) in enumerate(zip(all_words, all_pos)):
                words = []
                words.append(all_words[i - 1] if i > 0 else START)
                words.append(all_words[i - 2] if i > 1 else START)
                words.append(all_words[i])
                words.append(all_words[i + 1] if i + 1 < len(all_words) else None)
                words.append(all_words[i + 2] if i + 2 < len(all_words) else None)

                sparse_vec = self._dl.to_sparse(all_words, [prev_pos[-2], prev_pos[-1]], i)
                self._probs[str(([prev_pos[-2], prev_pos[-1]], words))] = \
                    self._model.predict_proba(sparse_vec)[0]
                prev_pos.append(pos)

    @staticmethod
    def _my_log(x):
        if x == 0:
            return -100
        if x == 1:
            return -0.001
        else:
            return np.log(x)

    def memm_tag(self, out_name="res_MEMM"):
        out_file = open(out_name, "wt")

        all_count = 0
        true_count = 0
        len_data = len(self._dl)
        for i, (all_pos, all_words) in enumerate(self._dl.data):
            if (100 * i / len_data) % 10 == 0:
                print(str((100 * i / len_data)) + "%")
            curr_pred = self._tagger.pred_viterbi(all_words, log=True)

            # print tp screen
            identical = sum([1 for p, l in zip(curr_pred, all_pos) if p == l])
            recall = str(int(identical / len(curr_pred) * 100))
            print("pred: " + str(curr_pred) + "\nlabel: " + str(all_pos) +
                  "\nrecall:\t" + str(identical) + "/" + str(len(curr_pred)) + "\t~" + recall + "%")

            # write to file
            for w, p in zip(all_words, curr_pred):
                out_file.write(w + "/" + p)
            out_file.write("\n")

            # calc recall
            for p, t in zip(all_pos, curr_pred):
                all_count += 1
                true_count += 1 if p == t else 0
            all_count += 1
        print(all_count, true_count, "\t~" + str(int(100*true_count/all_count)) + "%")
        out_file.close()


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print("input\t\tinput_file_name, modelname, feature_map_file, out_file_name")
    MEMMTagger(args[0], args[1], args[2]).memm_tag(out_name=args[3])

