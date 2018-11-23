import numpy as np
START = "START"


# --------------------- prob_func
# input:  word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=T/F
# output: best_score, back pointer

class ViterbiAlg:
    def __init__(self, pos_list, prob_func, delta=(0.2, 0.5, 0.3)):
        self._prob_func = prob_func
        self._delta = delta
        self._pos_list = pos_list
        self._num_pos = len(pos_list)
        self._pos_idx = {pos: i for i, pos in enumerate(self._pos_list)}

    @staticmethod
    def _my_log(x):
        if x == 0:
            return -100
        else:
            return np.log(x)

    def pred_viterbi(self, sequence, log=False):
        print("Viterbi - START...")
        print("Viterbi - INITIALIZATION...")
        # ------------ INITIALIZATION --------------
        len_seq = len(sequence) + 1
        base_score = self._my_log(0) if log else 0
        v_mx = [[[(base_score, (-1, self._pos_idx[START], self._pos_idx[START])) for _ in range(self._num_pos)] for _ in
                 range(self._num_pos)] for _ in range(len_seq)]
        bp = (-1, self._pos_idx[START], self._pos_idx[START])
        base_score = self._my_log(1) if log else 1
        v_mx[0][self._pos_idx[START]][self._pos_idx[START]] = (base_score, bp)

        print("Viterbi - FORWARD...")
        # ------- RECURSIVE STEP / FORWARD ---------
        print("Viterbi - forward: " + str(sequence) + "\nProgress:\t", end="")
        for i in range(1, len_seq):
            print(".", end="")
            for j, pos2 in enumerate(self._pos_list):
                for k, pos1 in enumerate(self._pos_list):
                    if pos1 == START:
                        score, bp = ((v_mx[i - 1][0][j][0] + self._my_log(0)) * 2 if log else 0), 0
                    else:
                        score, bp = self._max_and_bp(sequence, v_mx, i-1, pos2, pos1, log=log)
                    bp = (i - 1, bp, j)
                    v_mx[i][j][k] = (score, bp)
        print(" -- forward completed --")
        print("Viterbi - BACKWARDS...")
        # ------- REPRODUCTION / BACKWARDS ---------
        # find max and arg max at v_max[last_layer]
        max_val = self._my_log(0) if log else 0
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

    def _max_and_bp(self, sequence, v_mx, word_idx, pos2, pos1, log=False):
        # given a word w_n and pos2, pos1
        # we want to maximize w_n is pos1 coming after a pos2 word
        # scores = V(w_n-1, pos_i, pos2) * q(pos1| pos_i, pos2) * e(w_n| pos1)  i = 0..num_pos
        from_row = [v_mx[word_idx][i][self._pos_idx[pos2]][0] for i in range(self._num_pos)]

        max_score, argmax_score = self._prob_func(sequence, word_idx, from_row, pos2, pos1, log=log)
        return max_score, argmax_score
