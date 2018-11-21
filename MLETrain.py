import numpy as np
import sys
from loggers import PrintLogger

START = "START"
PREF = 3
SUFF = 4
CUT = 0.5


class MleEstimator:
    def __init__(self, source_file, num_prefix=120, num_suffix=200, delta=(0.2, 0.5, 0.3), gamma=(1, 1)):
        self._logger = PrintLogger("NLP-ass1")
        self._delta = delta
        self._gamma = gamma
        self._source = source_file
        self._num_prefix = num_prefix
        self._num_suffix = num_suffix
        # counters
        self._emission_count, self._transition_count, self._prefix_count, self._suffix_count = self._get_data()
        self._pos_list = list(set(list(self._transition_count[0].keys()) + [START]))
        self._num_pos = len(self._pos_list)
        self._pos_idx = {pos: i for i, pos in enumerate(self._pos_list)}
        # probabilities
        #self._emission, self._transition, self._prefix, self._suffix = self._calc_probabilities()

    def _get_data(self):
        self._logger.info("get-data - start")
        transition = {0: {}, 1: {}, 2: {}}
        t1 = START
        t2 = START
        emission = {}
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
                emission[(word, pos)] = emission.get((word, pos), 0) + 1             # count (word, POS)++
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
        return emission, transition, prefix, suffix

    def mle_count_to_txt(self, e_mle_path, q_mle_path):
        self._logger.info("writing e_mle...")
        out_e = open(e_mle_path, "wt")
        out_e.writelines([word + " " + pos + "\t" + str(count) + "\n" for (word, pos), count
                          in self._emission_count.items()])
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


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    mm = MleEstimator(input_file_name)
    mm.mle_count_to_txt(e_mle_filename, q_mle_filename)
