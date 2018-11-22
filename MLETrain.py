import numpy as np
import sys
from loggers import PrintLogger

START = "START"
SUFF = 4
CUT = 0.5


def my_log(x):
    if x == 0:
        return -100
    if x == 1:
        return -0.001
    else:
        return np.log(x)


def extract_words_pos_from_file(file):
    src_file = open(file, "rt")  # open file
    for line in src_file:
        seq = []
        label = []
        for w_p in line.split():  # break line to [.. (word, POS) ..]
            word, pos = w_p.rsplit("/", 1)
            seq.append(word)
            label.append(pos)
            return seq, label


def extract_e_from_file(e_file):
    src_file = open(e_file, "rt")  # open file
    emission_count = {}
    suffix_count = {}
    for line in src_file:
        format, count = line.split("\t")
        word, pos = format.split()
        if word[0] == '^':
            suff = word[1:]
            suffix_count[(suff, pos)] = int(count)
        else:
            emission_count[(word, pos)] = int(count)
    return emission_count, suffix_count


def extract_q_from_file(q_file):
    src_file = open(q_file, "rt")  # open file
    transition_count = {0: {}, 1: {}, 2: {}}
    for line in src_file:
        pos_seq, count = line.split("\t")
        pos_seq = tuple(pos_seq.split())
        if len(pos_seq) == 1:
            transition_count[len(pos_seq) - 1][pos_seq[0]] = int(count)
        else:
            transition_count[len(pos_seq)-1][pos_seq] = int(count)
    return transition_count

def extract_lambdas(file):
    src_file = open(file, "rt")  # open file
    transition_lambdas = []
    for line in src_file:
        transition_lambdas.append(float(line))
    return transition_lambdas

def calc_probabilities(transition_count, emission_count, suffix_count):
    # self._logger.info("calc-probabilities - start")
    transition_prob = {}

    # -------- EMISSION ----------
    # e(word| pos)

    #for (word, pos), w_p_count in emission_count.items():
     #   print(transition_count[0][pos])
    emission_prob = {(word, pos): ((1 - CUT) * w_p_count / transition_count[0][pos]) + CUT for (word, pos), w_p_count
                     in emission_count.items()}

    # --------- PREFIX -----------
    # given word [w_1, w_2 , ... , w_n-1, w_n]
    # e(w_n-1, w_n| pos)
    suffix_bi_prob = {(sufi, pos): ((1 - CUT) * s_p_count / transition_count[0][pos]) + CUT for (sufi, pos), s_p_count
                      in suffix_count.items()}

    # ------- TRANSITION ---------
    sum_words = np.sum(list(transition_count[0].values()))
    # sequence = [pos2, pos1, pos0]
    # q(pos0)
    transition_prob[0] = {pos: ((1 - CUT) * pos_count / sum_words) + CUT for pos, pos_count in
                          transition_count[0].items()}
    # q(pos0| pos1)
    transition_prob[1] = {(pos1, pos0): ((1 - CUT) * count / transition_count[0][pos1]) + CUT
                          for (pos1, pos0), count in transition_count[1].items()
                          if pos1 in transition_count[0]}
    # q(pos0| pos2, pos1)
    transition_prob[2] = {(pos2, pos1, pos0): ((1 - CUT) * count / transition_count[1][(pos2, pos1)]) + CUT
                          for (pos2, pos1, pos0), count in transition_count[2].items()
                          if (pos2, pos1) in transition_count[1]}
    # self._logger.info("calc-probabilities - end")
    return emission_prob, transition_prob, suffix_bi_prob


def emission(word_pos: tuple, emission, suffix, log=False):
    word, pos = word_pos
    # if there is a value e(word|pos)
    if (word, pos) in emission:
        return my_log(emission[word_pos]) if log else emission[word_pos]
    suf = word[-SUFF:]
    return my_log(suffix.get((suf, pos), 0)) if log else suffix.get((suf, pos), 0)


def transition(pos_sequence: tuple, transition, transition_lambdas, log=False):
    # break sequence
    pos0 = pos_sequence[-1]
    pos1 = pos_sequence[-2]
    pos2 = pos_sequence[-3] if len(pos_sequence) > 2 else None
    # calculate:   d1*q(pos0| pos2, pos1)   +   d2*q(pos0| pos1)   +   d3*q(pos0)
    tran_0 = transition_lambdas[0] * transition[0].get(pos0, 0)
    tran_1 = transition_lambdas[1] * transition[1].get((pos1, pos0), 0)
    tran_2 = transition_lambdas[2] * transition[2].get((pos2, pos1, pos0), 0) if pos2 else 0
    return my_log(tran_2 + tran_1 + tran_0) if log else tran_2 + tran_1 + tran_0


class MleExtractor:
    def __init__(self, e_mle_file, q_mle_file, extra_file):
        self._emission_count, self._suffix_count = extract_e_from_file(e_mle_file)
        self._transition_count = extract_q_from_file(q_mle_file)
        self._transition_lambdas = extract_lambdas(extra_file)
        self._emission, self._transition, self._suffix = calc_probabilities(self._transition_count, self._emission_count, self._suffix_count)
        self._pos_list = list(set(list(self._transition_count[0].keys()) + [START]))

    def prob_func(self, word_sequence, curr_word_idx, src_prob_row, prev_POS, curr_POS, log=True):
        # given a word w_n and pos2, pos1
        # we want to maximize w_n is pos1 coming after a pos2 word
        # scores = V(w_n-1, pos_i, pos2) * q(pos1| pos_i, pos2) * e(w_n| pos1)  i = 0..num_pos
        if log:
            if np.std(src_prob_row) > 0:
                good_chance = np.argsort(src_prob_row)[-8:]
            else:
                good_chance = list(range(len(src_prob_row)))
            scores = {
            i: src_prob_row[i] + transition((self._pos_list[i], prev_POS, curr_POS), self._transition, self._transition_lambdas, log=log) +
               emission((word_sequence[curr_word_idx], curr_POS), self._emission, self._suffix, log=log) for i in good_chance}
        else:
            scores = [src_prob_row[i] * transition((self._pos_list[i], prev_POS, curr_POS), self._transition, self._transition_lambdas, log=log) *
                      emission((word_sequence[curr_word_idx], curr_POS), self._emission, self._suffix, log=log) for i in
                      range(self._num_pos)]
        argmax_score, max_score = max(scores.items(), key=lambda x: x[1])
        # argmax_score = np.argmax(scores)
        return max_score, argmax_score



class MleEstimator:
    def __init__(self, source_file, num_prefix=120, num_suffix=200, delta=(0.2, 0.5, 0.3), gamma=(1, 1)):
        self._logger = PrintLogger("NLP-ass1")
        self._delta = delta
        self._gamma = gamma
        self._source = source_file
        self._num_prefix = num_prefix
        self._num_suffix = num_suffix
        # counters
        self._emission_count, self._transition_count, self._suffix_count = self._get_data()
        self._pos_list = list(set(list(self._transition_count[0].keys()) + [START]))
        self._num_pos = len(self._pos_list)
        self._pos_idx = {pos: i for i, pos in enumerate(self._pos_list)}
        # probabilities
        #self._emission, self._transition, self._prefix, self._suffix = self._calc_probabilities()

    def _get_data(self):
        self._logger.info("get-data - start")
        transition = {0: {}, 1: {}, 2: {}}
        emission = {}
        suffix = {}
        word_counter = 0
        src_file = open(self._source, "rt")                                          # open file
        for line in src_file:
            t1 = START
            t2 = START
            w_pos = []
            for w_p in line.split():                                                 # break line to [.. (word, POS) ..]
                word, pos = w_p.rsplit("/", 1)
                w_pos.append((word, pos))
            for i, (word, pos) in enumerate(w_pos):
                word_counter += 1
                # -------- EMISSION ----------
                emission[(word, pos)] = emission.get((word, pos), 0) + 1  # count (word, POS)++
                # --------- SUFFIX -----------
                if word_counter % 10 == 0:
                    suffix[(word[-SUFF:], pos)] = suffix.get((word[-SUFF:], pos), 0) + 1
                # ------- TRANSITION ---------
                transition[0][pos] = transition[0].get(pos, 0) + 1  # count(POS)
                transition[1][(t1, pos)] = transition[1].get((t1, pos), 0) + 1  # count(POS_1, POS_2)
                transition[2][(t2, t1, pos)] = transition[2].get((t2, t1, pos), 0) + 1  # count(POS_0, POS_1, POS_2)
                t2 = t1
                t1 = pos
        self._logger.info("get-data - end")
        return emission, transition, suffix

    def mle_count_to_txt(self, e_mle_path, q_mle_path):
        self._logger.info("writing e_mle...")
        out_e = open(e_mle_path, "wt")
        out_e.writelines([word + " " + pos + "\t" + str(count) + "\n" for (word, pos), count
                          in self._emission_count.items()])
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



if __name__ == "__main__":
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    mm = MleEstimator(input_file_name)
    mm.mle_count_to_txt(e_mle_filename, q_mle_filename)
