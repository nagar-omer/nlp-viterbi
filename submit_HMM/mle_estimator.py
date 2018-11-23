import numpy as np
from loggers import PrintLogger
from utils.viterbi import ViterbiAlg

START = "START"
PREF = 3
SUFF = 4
CUT = 0.5


class MleEstimator:
    def __init__(self, source_file, num_prefix=120, num_suffix=200, delta=(0.2, 0.5, 0.3)):
        self._logger = PrintLogger("NLP-ass1")
        self._delta = delta
        self._source = source_file
        self._num_prefix = num_prefix
        self._num_suffix = num_suffix
        # counters
        self._emmision_count, self._transition_count, self._prefix_count, self._suffix_count = self._get_data()
        self._pos_list = list(set(list(self._transition_count[0].keys()) + [START]))
        self._num_pos = len(self._pos_list)
        self._pos_idx = {pos: i for i, pos in enumerate(self._pos_list)}
        # probabilities
        self._emmision, self._transition, self._prefix, self._suffix = self._calc_probabilities()

    def _get_data(self):
        self._logger.info("get-data - start")
        transition = {0: {}, 1: {}, 2: {}}
        t1 = START
        t2 = START
        emmision = {}
        prefix = {}
        suffix = {}
        src_file = open(self._source, "rt")                                          # open file
        for line in src_file:
            # ---------- BREAK -----------
            w_pos = []
            for w_p in line.split():                                                 # break line to [.. (word, POS) ..]
                word, pos = w_p.rsplit("/", 1)
                w_pos.append((word, pos))
            for i, (word, pos) in enumerate(w_pos):
                # -------- EMISSION ----------
                emmision[(word, pos)] = emmision.get((word, pos), 0) + 1             # count (word, POS)++
                # --------- PREFIX -----------
                prefix[(word[:PREF], pos)] = prefix.get((word[:PREF], pos), 0) + 1    # count bigram prefixes
                suffix[(word[-SUFF:], pos)] = prefix.get((word[-SUFF:], pos), 0) + 1  # count bigram prefixes
                # ------- TRANSITION ---------
                transition[0][pos] = transition[0].get(pos, 0) + 1                      # count(POS)
                transition[1][(t1, pos)] = transition[1].get((t1, pos), 0) + 1          # count(POS_1, POS_2)
                transition[2][(t2, t1, pos)] = transition[2].get((t2, t1, pos), 0) + 1  # count(POS_0, POS_1, POS_2)
                t2 = t1
                t1 = pos
        prefix = {pre: pos for i, (pre, pos) in enumerate(sorted(prefix.items(), key=lambda x: -x[1]))
                  if i < self._num_prefix}
        suffix = {pre: pos for i, (pre, pos) in enumerate(sorted(suffix.items(), key=lambda x: -x[1]))
                  if i < self._num_suffix}
        # take K most common prefixes
        self._logger.info("get-data - end")
        return emmision, transition, prefix, suffix

    @staticmethod
    def _my_log(x):
        if x == 0:
            return -100
        if x == 1:
            return -0.001
        else:
            return np.log(x)

    def _calc_probabilities(self):
        self._logger.info("calc-probabilities - start")
        transition_prob = {}

        # -------- EMISSION ----------
        # e(word| pos)
        emmision_prob = {(word, pos): ((1-CUT) * w_p_count / self._transition_count[0][pos]) + CUT for (word, pos), w_p_count
                         in self._emmision_count.items()}

        # --------- PREFIX -----------
        # given word [w_1, w_2 , ... , w_n-1, w_n]
        # e(w_n-1, w_n| pos)
        prefix_bi_prob = {(pre, pos): ((1-CUT) * s_p_count / self._transition_count[0][pos]) + CUT for (pre, pos), s_p_count
                          in self._prefix_count.items()}
        suffix_bi_prob = {(sufi, pos): ((1-CUT) * s_p_count / self._transition_count[0][pos]) + CUT for (sufi, pos), s_p_count
                          in self._suffix_count.items()}

        # ------- TRANSITION ---------
        sum_words = np.sum(list(self._transition_count[0].values()))
        # sequence = [pos2, pos1, pos0]
        # q(pos0)
        transition_prob[0] = {pos: ((1-CUT) * pos_count/sum_words) + CUT for pos, pos_count in self._transition_count[0].items()}
        # q(pos0| pos1)
        transition_prob[1] = {(pos1, pos0): ((1-CUT) * count / self._transition_count[0][pos1]) + CUT
                              for (pos1, pos0), count in self._transition_count[1].items()
                              if pos1 in self._transition_count[0]}
        # q(pos0| pos2, pos1)
        transition_prob[2] = {(pos2, pos1, pos0): ((1-CUT) * count / self._transition_count[1][(pos2, pos1)]) + CUT
                              for (pos2, pos1, pos0), count in self._transition_count[2].items()
                              if (pos2, pos1) in self._transition_count[1]}
        self._logger.info("calc-probabilities - end")
        return emmision_prob, transition_prob, prefix_bi_prob, suffix_bi_prob

    def emmision(self, word_pos: tuple, log=False):
        # break
        word, pos = word_pos
        # if there is a value e(word| vec)
        if (word, pos) in self._emmision:
            return self._my_log(self._emmision[word_pos]) if log else self._emmision[word_pos]
        # if not then check if there is a value e(w_1, w_2| pos)
        pref = word[:PREF]
        if (pref, pos) in self._prefix:
            return self._my_log(self._prefix[(pref, pos)]) if log else self._prefix[(pref, pos)]
        # if not then check if there is a value e(w_n-1, w_n| pos)
        suf = word[-SUFF:]
        if (suf, pos) in self._suffix:
            return self._my_log(self._suffix[(suf, pos)]) if log else self._suffix[(suf, pos)]
        return self._my_log(0) if log else 0

    def transition(self, pos_sequence: tuple, log=False):
        # break sequence
        pos0 = pos_sequence[-1]
        pos1 = pos_sequence[-2]
        pos2 = pos_sequence[-3] if len(pos_sequence) > 2 else None
        # calculate:   d1*q(pos0| pos2, pos1)   +   d2*q(pos0| pos1)   +   d3*q(pos0)
        tran_0 = self._delta[0] * self._transition[0].get(pos0, 0)
        tran_1 = self._delta[1] * self._transition[1].get((pos1, pos0), 0)
        tran_2 = self._delta[2] * self._transition[2].get((pos2, pos1, pos0), 0) if pos2 else 0
        return self._my_log(tran_2 + tran_1 + tran_0) if log else tran_2 + tran_1 + tran_0

    def mle_count_to_txt(self, e_mle_path, q_mle_path):
        self._logger.info("writing e_mle...")
        out_e = open(e_mle_path, "wt")
        out_e.writelines([word + " " + pos + "\t" + str(count) + "\n" for (word, pos), count
                          in self._emmision_count.items()])
        out_e.writelines(["^" + pref + " " + pos + "\t" + str(count) + "\n" for (pref, pos), count
                          in self._prefix_count.items()])
        out_e.writelines(["^" + sufi + " " + pos + "\t" + str(count) + "\n" for (sufi, pos), count
                          in self._suffix_count.items()])
        out_e.close()
        self._logger.info("writing q_mle...")
        out_q = open(q_mle_path, "wt")
        out_q.writelines([pos + "\t" + str(count) + "\n" for pos, count in self._transition_count[0].items()])
        out_q.writelines([pos1 + " " + pos0 + "\t" + str(count) + "\n" for (pos1, pos0), count
                          in self._transition_count[1].items()])
        out_q.writelines([pos2 + " " + pos1 + " " + pos0 + "\t" + str(count) + "\n" for (pos2, pos1, pos0), count
                          in self._transition_count[2].items()])
        out_q.close()

    def prob_func(self, word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=True):
        if curr_POS == START:
            return -200 if log else 0, 0

        _0_3 = self._my_log(0.5) if log else 0.3
        # given a word w_n and pos2, pos1
        # we want to maximize w_n is pos1 coming after a pos2 word
        # scores = V(w_n-1, pos_i, pos2) * q(pos1| pos_i, pos2) * e(w_n| pos1)  i = 0..num_pos
        if log:
            if np.std(src_prob_row) > 0:
                good_chance = np.argsort(src_prob_row)[-8:]
            else:
                good_chance = list(range(len(src_prob_row)))
            scores = {i: src_prob_row[i] + self.transition((self._pos_list[i], prev_POS, curr_POS), log=log) +
                      self.emmision((word_sequence[curr_word_idx], curr_POS), log=log)
                      for ii, i in enumerate(good_chance) if ii < 3 or
                      self.emmision((word_sequence[curr_word_idx], curr_POS)) > _0_3}
        else:
            scores = [src_prob_row[i] * self.transition((self._pos_list[i], prev_POS, curr_POS), log=log) *
                      self.emmision((word_sequence[curr_word_idx], curr_POS), log=log) for i in range(self._num_pos)]
        argmax_score, max_score = max(scores.items(), key=lambda x: x[1])
        # argmax_score = np.argmax(scores)
        return max_score, argmax_score


if __name__ == "__main__":
    mm = MleEstimator("../data/ass1-tagger-train")
    src_file = open("../data/ass1-tagger-test", "rt")  # open file
    tagger = ViterbiAlg(mm._pos_list, mm.prob_func)
    for line in src_file:
        # ---------- BREAK -----------
        seq = []
        label = []
        for w_p in line.split():             # break line to [.. (word, POS) ..]
            word, pos = w_p.rsplit("/", 1)
            seq.append(word)
            label.append(pos)
        pred = tagger.pred_viterbi(seq, log=True)          # predict

        # print results
        identical = sum([1 for p, l in zip(pred, label) if p == l])
        recall = str(int(identical/len(pred) * 100))
        print("pred: " + str(pred) + "\nlabel: " + str(label) +
              "\nrecall:\t" + str(identical) + "/" + str(len(pred)) + "\t~" + recall + "%")

