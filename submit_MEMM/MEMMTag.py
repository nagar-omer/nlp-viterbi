import pickle
import numpy as np
from utils.data_loader import DataLoader
from utils.ftr_builders import TransitionFtr, EmmisionFtr, SuffixPrefix, CombinationsWordsPos, CostumeFtr
from utils.viterbi import ViterbiAlg

START = "START"
SIZE_FTR = 5000


class MEMMTagger:
    def __init__(self, to_pred, model_file, feature_map, out_name="greedy_pred"):
        self._model = pickle.load(open(model_file, "rb"))
        ftr_builders = [TransitionFtr(out_dim=SIZE_FTR), EmmisionFtr(out_dim=SIZE_FTR), SuffixPrefix(out_dim=SIZE_FTR),
                        CombinationsWordsPos(out_dim=SIZE_FTR), CostumeFtr()]
        self._dl = DataLoader(to_pred, feature_map, ftr_builders)
        self._label_list = self._dl.label_list + [START]
        self._label_to_idx = {label: i for i, label in enumerate(self._label_list)}
        self._tagger = ViterbiAlg(self._label_list, self._prob_func)

    # --------------------- prob_func
    # input:  word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=T/F
    # output: best_score, back pointer
    def _prob_func(self, word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=True):
        if curr_POS == START:
            return self._my_log(0) if log else 0, -1
        av = abs(np.average(src_prob_row) / 2)
        good_chance = [idx for idx, val in enumerate(src_prob_row) if abs(val) > av]
        scores = {}
        for prev_prev_pos in good_chance:
            # if (log and src_prob_row[prev_prev_pos] < self._my_log(0)) or \
            #         (not log and src_prob_row[prev_prev_pos] < 0.001):
            #     scores[prev_prev_pos] = self._my_log(0) if log else 0
            #     continue
            sparse_vec = self._dl.to_sparse(word_sequence, [prev_prev_pos, prev_POS], curr_word_idx)
            scores[prev_prev_pos] = self._model.predict_proba(sparse_vec)[0][self._label_to_idx[curr_POS]]
        back_pointer, mid_score = max(scores.items(), key=lambda x: x[1])
        best_score = src_prob_row[back_pointer] + self._my_log(mid_score) if log else \
            src_prob_row[back_pointer] * mid_score  # TODO fix calculation
        return best_score, back_pointer

    @staticmethod
    def _my_log(x):
        if x == 0:
            return -100
        else:
            return np.log(x)

    def memm_tag(self):
        all_count = 0
        true_count = 0
        len_data = len(self._dl)
        for i, (all_pos, all_words) in enumerate(self._dl.data):
            if (100 * i / len_data) % 10 == 0:
                print(str((100 * i / len_data)) + "%")
            curr_pred = self._tagger.pred_viterbi(all_words, log=True)
            for p, t in zip(all_pos, curr_pred):
                print(p, t)
                all_count += 1
                true_count += 1 if p == t else 0
            all_count += 1
        print(all_count, true_count, "\t~" + str(int(100*true_count/all_count)) + "%")


if __name__ == "__main__":
    MEMMTagger("../data/ass1-tagger-test", "model_file", "map_file").memm_tag()

