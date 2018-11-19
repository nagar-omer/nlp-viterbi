from copy import deepcopy
START = "START"


class FeatureBuilder:
    def __init__(self, out_dim):
        self._out_dim = out_dim
        self._features_list = []

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def name(self):
        raise NotImplementedError()

    def learn_example(self, all_words, prev_pos, i, pos):
        raise NotImplementedError()

    def to_vec(self, all_words, prev_pos, i, pos):
        raise NotImplementedError()

    def get_ftrs(self):
        return self._features_list

    def ftr_str(self, all_words, prev_pos, i, pos):
        raise NotImplementedError()


class TransitionFtr(FeatureBuilder):
    def __init__(self, out_dim=500):
        super(TransitionFtr, self).__init__(0)
        self._goal_dim = out_dim
        self._transition = {}
        self._updated = False
        self._selected = {}

    def learn_example(self, all_words, prev_pos, i, pos):
        self._updated = False
        pos1 = prev_pos[-1] if len(prev_pos) > 0 else START
        pos2 = prev_pos[-2] if len(prev_pos) > 1 else START
        self._transition[(pos1, pos)] = self._transition.get((pos1, -1, pos), 0) + 1  # count(POS_1, POS_2)
        self._transition[(pos2, pos1, pos)] = self._transition.get((pos2, pos1, -2, pos), 0) + 1  # count(POS_0, POS_1,POS2)

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[:-1]: 0 for i, (seq, count) in
                          enumerate(sorted(self._transition.items(), key=lambda x: -x[1])) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def to_vec(self, all_words, prev_pos, i, pos):
        self._update()
        pos1 = prev_pos[-1] if len(prev_pos) > 0 else START
        pos2 = prev_pos[-2] if len(prev_pos) > 1 else START
        ftr_dict = deepcopy(self._selected)
        if (pos1, -1) in self._selected:
            ftr_dict[(pos1, -1)] = 1  # count(POS_1, POS_2)
        if (pos2, pos1, -2) in self._selected:
            ftr_dict[(pos2, pos1, -2)] = 1  # count(POS_0, POS_1,POS2)
        return ftr_dict

    def ftr_str(self, all_words, prev_pos, i, pos):
        self._update()
        pos1 = prev_pos[-1] if len(prev_pos) > 0 else START
        pos2 = prev_pos[-2] if len(prev_pos) > 1 else START
        return " prev_tag=" + pos1 + " prev_prev_tag=" + pos2

    def get_ftrs(self):
        self._update()
        return self._features_list

    @property
    def name(self):
        return "prev_tokens"


class EmmisionFtr(FeatureBuilder):
    def __init__(self, out_dim=500):
        super(EmmisionFtr, self).__init__(0)
        self._goal_dim = out_dim
        self._emmision = {}
        self._updated = False
        self._selected = {}

    def learn_example(self, all_words, prev_pos, i, pos):
        self._updated = False
        self._emmision[(all_words[i], 0, pos)] = \
            self._emmision.get((all_words[i], 0, pos), 0) + 1  # count (word, POS)++
        if len(all_words) > i + 1:
            self._emmision[(all_words[i+1], 1, pos)] = \
                self._emmision.get((all_words[i+1], 1, pos), 0) + 1  # count (word, POS)++
        if i > 0:
            self._emmision[(all_words[i-1], -1, pos)] = \
                self._emmision.get((all_words[i-1], -1, pos), 0) + 1  # count(word, POS)++
        if i > 1:
            self._emmision[(all_words[i-2], -2, pos)] = \
                self._emmision.get((all_words[i-2], -2, pos), 0) + 1  # count (word, POS)++

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[:-1]: 0 for i, (seq, count) in
                          enumerate(sorted(self._emmision.items(), key=lambda x: -x[1])) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def to_vec(self, all_words, prev_pos, i, pos):
        self._update()
        ftr_dict = deepcopy(self._selected)
        if len(all_words) > i + 1 and (all_words[i-1], -1) in self._selected:
            ftr_dict[(all_words[i+1], 1)] = 1
        if (all_words[i], 0) in self._selected:
            ftr_dict[(all_words[i], 0)] = 1
        if i > 0 and (all_words[i-1], -1) in self._selected:
            ftr_dict[(all_words[i-1], -1)] = 1
        if i > 1 and (all_words[i-2], -2) in self._selected:
            ftr_dict[(all_words[i-2], -2)] = 1
        return ftr_dict

    def ftr_str(self, all_words, prev_pos, i, pos):
        self._update()
        ftr_str = ""
        if i > 1:
            ftr_str += " prev_prev_word=" + all_words[i-2]
        if i > 0:
            ftr_str += " prev_word=" + all_words[i-1]
        ftr_str += " curr_word=" + all_words[i]
        if len(all_words) > i + 1:
            ftr_str += " next_word=" + all_words[i+1]
        return ftr_str

    def get_ftrs(self):
        self._update()
        return self._features_list

    @property
    def name(self):
        return "local_words"


class SuffixPrefix(FeatureBuilder):
    def __init__(self, out_dim=500, pref_size=3, suff_size=3):
        super(SuffixPrefix, self).__init__(0)
        self._pref_size = pref_size
        self._suff_size = suff_size
        self._goal_dim = out_dim
        self._pre_suf_fix = {}
        self._updated = False
        self._selected = {}
        self._suf_token = "*S"
        self._pref_token = "P*"

    def learn_example(self, all_words, prev_pos, i, pos):
        self._updated = False
        word = all_words[i]
        self._pre_suf_fix[(self._pref_token, word[:self._pref_size], 0, pos)] = \
            self._pre_suf_fix.get((self._pref_token, word[:self._pref_size], 0, pos), 0) + 1  # count bigram prefixes
        self._pre_suf_fix[(self._suf_token, word[-self._suff_size:], 0,  pos)] = \
            self._pre_suf_fix.get((self._suf_token, word[-self._suff_size:], 0, pos), 0) + 1  # count bigram prefixes
        if len(all_words) > i + 1:
            word = all_words[i+1]
            self._pre_suf_fix[(self._pref_token, word[:self._pref_size], 1, pos)] = \
                self._pre_suf_fix.get((self._pref_token, word[:self._pref_size], 1, pos),
                                      0) + 1  # count bigram prefixes
            self._pre_suf_fix[(self._suf_token, word[-self._suff_size:], 1, pos)] = \
                self._pre_suf_fix.get((self._suf_token, word[-self._suff_size:], 1, pos),
                                      0) + 1  # count bigram prefixes
        if i > 0:
            word = all_words[i-1]
            self._pre_suf_fix[(self._pref_token, word[:self._pref_size], -1, pos)] = \
                self._pre_suf_fix.get((self._pref_token, word[:self._pref_size], -1, pos),
                                      0) + 1  # count bigram prefixes
            self._pre_suf_fix[(self._suf_token, word[-self._suff_size:], -1, pos)] = \
                self._pre_suf_fix.get((self._suf_token, word[-self._suff_size:], -1, pos),
                                      0) + 1  # count bigram prefixes

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[:-1]: 0 for i, (seq, count) in
                          enumerate(sorted(self._pre_suf_fix.items(), key=lambda x: -x[1])) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def to_vec(self, all_words, prev_pos, i, pos):
        self._update()
        ftr_dict = deepcopy(self._selected)
        word = all_words[i]
        if (self._pref_token, word[:self._pref_size], 0) in self._selected:
            ftr_dict[(self._pref_token, word[:self._pref_size], 0)] = 1
        if (self._suf_token, word[-self._suff_size:], 0) in self._selected:
            ftr_dict[(self._suf_token, word[-self._suff_size:], 0)] = 1

        if len(all_words) > i + 1:
            word = all_words[i+1]
            if (self._pref_token, word[:self._pref_size], 1) in self._selected:
                ftr_dict[(self._pref_token, word[:self._pref_size], 1)] = 1
            if (self._suf_token, word[-self._suff_size:], 1) in self._selected:
                ftr_dict[(self._suf_token, word[-self._suff_size:], 1)] = 1
        if i > 0:
            word = all_words[i-1]
            if (self._pref_token, word[:self._pref_size], -1) in self._selected:
                ftr_dict[(self._pref_token, word[:self._pref_size], -1)] = 1
            if (self._suf_token, word[-self._suff_size:], -1) in self._selected:
                ftr_dict[(self._suf_token, word[-self._suff_size:], -1)] = 1
        return ftr_dict

    def ftr_str(self, all_words, prev_pos, i, pos):
        ftr_str = ""
        if i > 0:
            ftr_str += " prev_suf=" + all_words[i-1][:self._pref_size]
            ftr_str += " prev_pref=" + all_words[i-1][-self._suff_size:]
        ftr_str += " curr_suf=" + all_words[i][:self._pref_size]
        ftr_str += " curr_pref=" + all_words[i][-self._suff_size:]
        if len(all_words) > i + 1:
            ftr_str += " prev_suf=" + all_words[i+1][:self._pref_size]
            ftr_str += " prev_pref=" + all_words[i+1][-self._suff_size:]
        return ftr_str

    def get_ftrs(self):
        self._update()
        return self._features_list

    @property
    def name(self):
        return "local_prev_suf"


class CombinationsWordsPos(FeatureBuilder):
    def __init__(self, out_dim=500, pref_size=3, suff_size=3):
        super(CombinationsWordsPos, self).__init__(0)
        self._pref_size = pref_size
        self._suff_size = suff_size
        self._goal_dim = out_dim
        self._combinations = {}
        self._updated = False
        self._selected = {}
        self._suf_token = "*S"
        self._pref_token = "P*"

    def learn_example(self, all_words, prev_pos, i, pos):
        self._updated = False
        prev_pos = prev_pos[-1] if len(prev_pos) > 0 else START
        word = all_words[i]
        self._combinations[(self._pref_token, word[:self._pref_size], prev_pos, 0, pos)] = \
            self._combinations.get((self._pref_token, word[:self._pref_size], prev_pos[-1], 0, pos), 0) + 1  # count bigram prefixes
        self._combinations[(self._suf_token, word[-self._suff_size:], prev_pos, 0, pos)] = \
            self._combinations.get((self._suf_token, word[-self._suff_size:], prev_pos[-1], 0, pos), 0) + 1  # count bigram prefixes
        if len(all_words) > i + 1:
            word = all_words[i+1]
            self._combinations[(self._pref_token, word[:self._pref_size], prev_pos, 1, pos)] = \
                self._combinations.get((self._pref_token, word[:self._pref_size], prev_pos, 1, pos),
                                       0) + 1  # count bigram prefixes
            self._combinations[(self._suf_token, word[-self._suff_size:], prev_pos, 1, pos)] = \
                self._combinations.get((self._suf_token, word[-self._suff_size:], prev_pos, 1, pos),
                                       0) + 1  # count bigram prefixes
        if i > 0:
            word = all_words[i-1]
            self._combinations[(self._pref_token, word[:self._pref_size], prev_pos, -1, pos)] = \
                self._combinations.get((self._pref_token, word[:self._pref_size], prev_pos, -1, pos),
                                       0) + 1  # count bigram prefixes
            self._combinations[(self._suf_token, word[-self._suff_size:], prev_pos, -1, pos)] = \
                self._combinations.get((self._suf_token, word[-self._suff_size:], prev_pos, -1, pos),
                                       0) + 1  # count bigram prefixes

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[:-1]: 0 for i, (seq, count) in
                          enumerate(sorted(self._combinations.items(), key=lambda x: -x[1])) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def to_vec(self, all_words, prev_pos, i, pos):
        self._update()
        prev_pos = prev_pos[-1] if len(prev_pos) > 0 else START
        ftr_dict = deepcopy(self._selected)
        word = all_words[i]
        if (self._pref_token, word[:self._pref_size], prev_pos, 0) in self._selected:
            ftr_dict[(self._pref_token, word[:self._pref_size], prev_pos, 0)] = 1
        if (self._suf_token, word[-self._suff_size:], prev_pos, 0) in self._selected:
            ftr_dict[(self._suf_token, word[-self._suff_size:], prev_pos, 0)] = 1

        if len(all_words) > i + 1:
            word = all_words[i+1]
            if (self._pref_token, word[:self._pref_size], prev_pos, 1) in self._selected:
                ftr_dict[(self._pref_token, word[:self._pref_size], prev_pos, 1)] = 1
            if (self._suf_token, word[-self._suff_size:], prev_pos, 1) in self._selected:
                ftr_dict[(self._suf_token, word[-self._suff_size:], prev_pos, 1)] = 1
        if i > 0:
            word = all_words[i-1]
            if (self._pref_token, word[:self._pref_size], prev_pos, -1) in self._selected:
                ftr_dict[(self._pref_token, word[:self._pref_size], prev_pos, -1)] = 1
            if (self._suf_token, word[-self._suff_size:], prev_pos, -1) in self._selected:
                ftr_dict[(self._suf_token, word[-self._suff_size:], prev_pos, -1)] = 1
        return ftr_dict

    def ftr_str(self, all_words, prev_pos, i, pos):
        ftr_str = ""
        if i > 0:
            ftr_str += " prev_suf&prev_pos=" + str((all_words[i-1][:self._pref_size], prev_pos[-1]))
            ftr_str += " prev_pref&prev_pos=" + str((all_words[i-1][-self._suff_size:], prev_pos[-1]))
        ftr_str += " curr_suf&prev_pos=" + str((all_words[i][:self._pref_size], prev_pos[-1]))
        ftr_str += " curr_pref&prev_pos=" + str((all_words[i][-self._suff_size:], prev_pos[-1]))
        if len(all_words) > i + 1:
            ftr_str += " next_suf&prev_pos=" + str((all_words[i+1][:self._pref_size], prev_pos[-1]))
            ftr_str += " next_pref&prev_pos=" + str((all_words[i+1][-self._suff_size:], prev_pos[-1]))
        return ftr_str

    def get_ftrs(self):
        self._update()
        return self._features_list

    @property
    def name(self):
        return "comb_word_pos"