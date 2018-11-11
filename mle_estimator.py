import numpy as np
from loggers import PrintLogger

START = "START"


class MleEstimator:
    def __init__(self, source_file, num_suffix=100, delta=(0.34, 0.33, 0.33)):
        self._logger = PrintLogger("NLP-ass1")
        self._delta = delta
        self._source = source_file
        self._num_suffix = num_suffix
        # counters
        self._emmision_count, self._transition_count, self._bigram_suffix_count, self._unigram_suffix_count = \
            self._get_data()
        self._pos_list = list(set(list(self._transition_count[0].keys()) + [START] ))
        self._num_pos = len(self._pos_list)
        self._pos_idx = {pos: i for i, pos in enumerate(self._pos_list)}
        # probabilities
        self._emmision, self._transition, self._suffix_bi, self._suffix_uni = self._calc_probabilities()

    def _get_data(self):
        self._logger.info("get-data - start")
        transition = {0: {}, 1: {}, 2: {}}
        t1 = START
        t2 = START
        emmision = {}
        suffix_bi = {}
        suffix_uni = {}
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
                # --------- SUFFIX -----------
                if len(word) > 1:
                    suffix_bi[(word[-2:], pos)] = suffix_bi.get((word[-2:], pos), 0) + 1     # count bigram suffixes
                    suffix_uni[(word[-1:], pos)] = suffix_uni.get((word[-1:], pos), 0) + 1   # count unigram suffixes
                # ------- TRANSITION ---------
                transition[0][pos] = transition[0].get(pos, 0) + 1                      # count(POS)
                transition[1][(t1, pos)] = transition[1].get((t1, pos), 0) + 1          # count(POS_1, POS_2)
                transition[2][(t2, t1, pos)] = transition[2].get((t2, t1, pos), 0) + 1  # count(POS_0, POS_1, POS_2)
                t2 = t1
                t1 = pos
        suffix_bi = {sufi: pos for i, (sufi, pos) in enumerate(sorted(suffix_bi.items(), key=lambda x: -x[1]))
                     if i < self._num_suffix}
        # take K most common suffixes
        self._logger.info("get-data - end")
        return emmision, transition, suffix_bi, suffix_uni

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
        emmision_prob = {(word, pos): w_p_count / self._transition_count[0][pos] for (word, pos), w_p_count
                         in self._emmision_count.items()}

        # --------- SUFFIX -----------
        # given word [w_1, w_2 , ... , w_n-1, w_n]
        # e(w_n-1, w_n| pos)
        suffix_bi_prob = {(sufi, pos): s_p_count / self._transition_count[0][pos] for (sufi, pos), s_p_count
                          in self._bigram_suffix_count.items()}
        # e(w_n| pos)
        suffix_uni_prob = {(sufi, pos): s_p_count / self._transition_count[0][pos] for (sufi, pos), s_p_count
                           in self._unigram_suffix_count.items()}

        # ------- TRANSITION ---------
        sum_words = np.sum(list(self._transition_count[0].values()))
        # sequence = [pos2, pos1, pos0]
        # q(pos0)
        transition_prob[0] = {pos: pos_count/sum_words for pos, pos_count in self._transition_count[0].items()}
        # q(pos0| pos1)
        transition_prob[1] = {(pos1, pos0): count / self._transition_count[0][pos1]
                              for (pos1, pos0), count in self._transition_count[1].items()
                              if pos1 in self._transition_count[0]}
        # q(pos0| pos2, pos1)
        transition_prob[2] = {(pos2, pos1, pos0): count / self._transition_count[1][(pos2, pos1)]
                              for (pos2, pos1, pos0), count in self._transition_count[2].items()
                              if (pos2, pos1) in self._transition_count[1]}
        self._logger.info("calc-probabilities - end")
        return emmision_prob, transition_prob, suffix_bi_prob, suffix_uni_prob

    def emmision(self, word_pos: tuple, log=False):
        # break
        word, pos = word_pos
        # if there is a value e(word| vec)
        if (word, pos) in self._emmision:
            return self._my_log(self._emmision[word_pos]) if log else self._emmision[word_pos]
        # if not then check if there is a value e(w_n-1, w_n| pos)
        bigram = word[-2:]
        if (bigram, pos) in self._suffix_bi:
            return self._my_log(self._suffix_bi[(bigram, pos)]) if log else self._suffix_bi[(bigram, pos)]
        # if not then check if there is a value e(w_n| pos)
        unigram = word[-1:]
        if (bigram, pos) in self._suffix_uni:
            return self._my_log(self._suffix_uni[(unigram, pos)]) if log else self._suffix_uni[(unigram, pos)]
        # else we have a problem!!
        else:
            # print("ERROR: can't generate emmision for - e(" + word + "| " + pos + ")")
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
        out_e.writelines(["^" + sufi + " " + pos + "\t" + str(count) + "\n" for (sufi, pos), count
                          in self._bigram_suffix_count.items()])
        out_e.writelines(["^" + sufi + " " + pos + "\t" + str(count) + "\n" for (sufi, pos), count
                          in self._unigram_suffix_count.items()])
        out_e.close()
        self._logger.info("writing q_mle...")
        out_q = open(q_mle_path, "wt")
        out_q.writelines([pos + "\t" + str(count) + "\n" for pos, count in self._transition_count[0].items()])
        out_q.writelines([pos1 + " " + pos0 + "\t" + str(count) + "\n" for (pos1, pos0), count
                          in self._transition_count[1].items()])
        out_q.writelines([pos2 + " " + pos1 + " " + pos0 + "\t" + str(count) + "\n" for (pos2, pos1, pos0), count
                          in self._transition_count[2].items()])
        out_q.close()

    def pred_viterbi(self, sequence, log=False):
        self._logger.info("Viterbi - START...")
        self._logger.info("Viterbi - INITIALIZATION...")
        # ------------ INITIALIZATION --------------
        len_seq = len(sequence) + 1
        base_score = self._my_log(0) if log else 0
        v_mx = [[[(base_score, (-1, self._pos_idx[START], self._pos_idx[START])) for _ in range(self._num_pos)] for _ in
                 range(self._num_pos)] for _ in range(len_seq)]
        bp = (-1, self._pos_idx[START], self._pos_idx[START])
        base_score = self._my_log(1) if log else 1
        v_mx[0][self._pos_idx[START]][self._pos_idx[START]] = (base_score, bp)

        self._logger.info("Viterbi - FORWARD...")
        # ------- RECURSIVE STEP / FORWARD ---------
        print("Viterbi - forward: " + str(sequence) + "\nProgress:          ", end="")
        for i in range(1, len_seq):
            print("." * (len(sequence[i-1]) + 3) + "|", end="")
            for j, pos2 in enumerate(self._pos_list):
                for k, pos1 in enumerate(self._pos_list):
                    score, bp = self._max_and_bp(v_mx, i, sequence[i - 1], j, pos2, pos1, log=log)
                    bp = (i - 1, bp, j)
                    v_mx[i][j][k] = (score, bp)
        print(" -- forward completed --")
        self._logger.info("Viterbi - BACKWARDS...")
        # ------- REPRODUCTION / BACKWARDS ---------
        # find max and arg max at v_max[last_layer]
        max_val = 0
        max_i = 0
        max_j = 0
        for i in range(self._num_pos):
            for j in range(self._num_pos):
                if v_mx[len_seq - 1][i][j][0] > max_val:
                    max_val = v_mx[len_seq - 1][i][j][0]
                    max_i = i
                    max_j = j
        # reconstruct Part Of Speech
        prediction = [self._pos_list[max_i], self._pos_list[max_j]]
        ps = v_mx[len_seq - 1][max_i][max_j][1]
        for word_idx in range(len_seq - 1, 0, -1):
            curr_pos = self._pos_list[ps[1]]
            if curr_pos == START:
                break
            prediction = [curr_pos] + prediction
            ps = v_mx[ps[0]][ps[1]][ps[2]][1]
        return prediction

    def _max_and_bp(self, v_mx, word_idx, word, pos2_idx, pos2, pos1, log=False):
        # given a word w_n and pos2, pos1
        # we want to maximize w_n is pos1 coming after a pos2 word
        # scores = V(w_n-1, pos_i, pos2) * q(pos1| pos_i, pos2) * e(w_n| pos1)  i = 0..num_pos
        if log:
            scores = [v_mx[word_idx - 1][i][pos2_idx][0] + self.transition((self._pos_list[i], pos2, pos1), log=log) +
                      self.emmision((word, pos1), log=log) for i in range(self._num_pos)]
        else:
            scores = [v_mx[word_idx-1][i][pos2_idx][0] * self.transition((self._pos_list[i], pos2, pos1), log=log) *
                      self.emmision((word, pos1), log=log) for i in range(self._num_pos)]
        max_score = np.max(scores)
        argmax_score = np.argmax(scores)
        return max_score, argmax_score


if __name__ == "__main__":
    mm = MleEstimator("data/ass1-tagger-train")
    src_file = open("data/ass1-tagger-test", "rt")  # open file
    for line in src_file:
        # ---------- BREAK -----------
        seq = []
        label = []
        for w_p in line.split():             # break line to [.. (word, POS) ..]
            word, pos = w_p.rsplit("/", 1)
            seq.append(word)
            label.append(pos)
        pred = mm.pred_viterbi(seq, log=True)          # predict

        # print results
        identical = sum([1 for p, l in zip(pred, label) if p == l])
        recall = str(int(identical/len(pred) * 100))
        print("pred: " + str(pred) + "\nlabel: " + str(label) +
              "\nrecall:\t" + str(identical) + "/" + str(len(pred)) + "\t~" + recall + "%")

