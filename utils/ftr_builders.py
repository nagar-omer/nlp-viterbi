from copy import deepcopy
START = "START"


class FeatureBuilder:
    def __init__(self, out_dim):
        self._features_list = []
        self._selected = {}

    @property
    def name(self):
        raise NotImplementedError()

    def learn_example(self, label, ftr_dict):
        raise NotImplementedError()

    def get_ftrs(self):
        return self._features_list

    def ftr_str(self, all_words, prev_pos, i):
        raise NotImplementedError()


class TransitionFtr(FeatureBuilder):
    def __init__(self, out_dim=2000):
        super(TransitionFtr, self).__init__(0)
        self._goal_dim = out_dim
        self._transition = {}
        self._updated = False
        self._selected = {}

    def learn_example(self, pos, example):
        self._updated = False
        key_p = ("prev_tag=" + example["prev_tag"], pos)
        key_pp = ("prev_prev_tag=" + example["prev_prev_tag"], pos)
        self._transition[key_p] = self._transition.get(key_p, 0) + 1  # count(POS_1, POS_2)
        self._transition[key_pp] = self._transition.get(key_pp, 0) + 1  # count(POS_0, POS_1,POS2)

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[0]: 0 for i, (seq, count) in enumerate(
            sorted(self._transition.items(), key=lambda x: (-x[1], x[0]))) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def ftr_str(self, all_words, prev_pos, i):
        self._update()
        pos1 = prev_pos[-1] if len(prev_pos) > 0 else START
        pos2 = prev_pos[-2] if len(prev_pos) > 1 else START
        return " prev_tag=" + pos1 + " prev_prev_tag=" + str((pos2, pos1)).replace(" ", "")

    def get_ftrs(self):
        self._update()
        return self._features_list

    @property
    def name(self):
        return "prev_tokens"


class EmmisionFtr(FeatureBuilder):
    def __init__(self, out_dim=2000):
        super(EmmisionFtr, self).__init__(0)
        self._goal_dim = out_dim
        self._emmision = {}
        self._updated = False
        self._selected = {}

    def learn_example(self, pos, example):
        pp_word_key = ("prev_prev_word=" + example["prev_prev_word"], pos) if "prev_prev_word" in example else None
        p_word_key = ("prev_word=" + example["prev_word"], pos) if "prev_word" in example else None
        c_word_key = ("curr_word=" + example["curr_word"], pos)
        n_word_key = ("next_word=" + example["next_word"], pos) if "next_word" in example else None

        self._emmision[c_word_key] = self._emmision.get(c_word_key, 0) + 1  # count (word, POS)++
        if n_word_key:
            self._emmision[n_word_key] = self._emmision.get(n_word_key, 0) + 1  # count (word, POS)++
        if p_word_key:
            self._emmision[p_word_key] = self._emmision.get(p_word_key, 0) + 1  # count(word, POS)++
        if pp_word_key:
            self._emmision[pp_word_key] = self._emmision.get(pp_word_key, 0) + 1  # count (word, POS)++

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[0]: 0 for i, (seq, count) in
                          enumerate(sorted(self._emmision.items(), key=lambda x: (-x[1], x[0]))) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def ftr_str(self, all_words, prev_pos, i):
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
    def __init__(self, out_dim=2000, pref_size=3, suff_size=3):
        super(SuffixPrefix, self).__init__(0)
        self._pref_size = pref_size
        self._suff_size = suff_size
        self._goal_dim = out_dim
        self._pre_suf_fix = {}
        self._updated = False
        self._selected = {}

    def learn_example(self, pos, example):
        self._updated = False
        p_suf_key = ("prev_suf=" + example["prev_suf"], pos) if "prev_suf" in example else None
        p_pref_key = ("prev_pref=" + example["prev_pref"], pos) if "prev_pref" in example else None
        c_suf_key = ("curr_suf=" + example["curr_suf"], pos)
        c_pref_key = ("curr_pref=" + example["curr_pref"], pos)
        n_suf_key = ("next_suf=" + + example["next_suf"], pos) if "next_suf" in example else None
        n_pref_key = ("next_pref=" + + example["next_pref"], pos) if "next_pref" in example else None

        self._pre_suf_fix[c_pref_key] = self._pre_suf_fix.get(c_pref_key, 0) + 1  # count bigram prefixes
        self._pre_suf_fix[c_suf_key] = self._pre_suf_fix.get(c_suf_key, 0) + 1  # count bigram prefixes
        if p_suf_key and p_pref_key:
            self._pre_suf_fix[p_suf_key] = self._pre_suf_fix.get(p_suf_key, 0) + 1  # count bigram prefixes
            self._pre_suf_fix[p_pref_key] = self._pre_suf_fix.get(p_pref_key, 0) + 1  # count bigram prefixes
        if n_suf_key and n_pref_key:
            self._pre_suf_fix[n_suf_key] = self._pre_suf_fix.get(n_suf_key, 0) + 1  # count bigram prefixes
            self._pre_suf_fix[n_pref_key] = self._pre_suf_fix.get(n_pref_key, 0) + 1  # count bigram prefixes

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[0]: 0 for i, (seq, count) in enumerate(
            sorted(self._pre_suf_fix.items(), key=lambda x: (-x[1], x[0]))) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def ftr_str(self, all_words, prev_pos, i):
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
    def __init__(self, out_dim=2000, pref_size=3, suff_size=3):
        super(CombinationsWordsPos, self).__init__(0)
        self._pref_size = pref_size
        self._suff_size = suff_size
        self._goal_dim = out_dim
        self._combinations = {}
        self._updated = False
        self._selected = {}

    def learn_example(self, pos, example):
        self._updated = False
        p_suf_key = ("prev_suf&prev_pos=" + example["prev_suf&prev_pos"], pos) if "prev_suf&prev_pos" in \
                                                                                  example else None
        p_pref_key = ("prev_pref&prev_pos=" + example["prev_pref&prev_pos"], pos) if "prev_pref&prev_pos" in \
                                                                                     example else None
        c_suf_key = ("curr_suf&prev_pos=" + example["curr_suf&prev_pos"], pos)
        c_pref_key = ("curr_suf&prev_pos=" + example["curr_suf&prev_pos"], pos)
        n_suf_key = ("next_suf&prev_pos=" + example["next_suf&prev_pos"], pos) if "next_suf&prev_pos" in \
                                                                                  example else None
        n_pref_key = ("next_suf&prev_pos=" + example["next_suf&prev_pos"], pos) if "next_suf&prev_pos" in \
                                                                                   example else None

        self._combinations[c_suf_key] = self._combinations.get(c_suf_key, 0) + 1  # count bigram prefixes
        self._combinations[c_pref_key] = self._combinations.get(c_pref_key, 0) + 1  # count bigram prefixes
        if n_pref_key and n_suf_key:
            self._combinations[n_suf_key] = self._combinations.get(n_suf_key, 0) + 1  # count bigram prefixes
            self._combinations[n_pref_key] = self._combinations.get(n_pref_key, 0) + 1  # count bigram prefixes
        if p_suf_key and p_pref_key:
            self._combinations[p_pref_key] = self._combinations.get(p_pref_key, 0) + 1  # count bigram prefixes
            self._combinations[p_suf_key] = self._combinations.get(p_suf_key, 0) + 1  # count bigram prefixes

    def _update(self):
        if self._updated:
            return
        self._selected = {seq[0]: 0 for i, (seq, count) in enumerate(
            sorted(self._combinations.items(), key=lambda x: (-x[1], x[0]))) if i < self._goal_dim}
        self._features_list = list(self._selected.keys())
        self._out_dim = len(self._features_list)
        self._updated = True

    def ftr_str(self, all_words, prev_pos, i):
        pos1 = prev_pos[-1] if len(prev_pos) > 0 else START
        ftr_str = ""
        if i > 0:
            ftr_str += " prev_suf&prev_pos=" + str((all_words[i-1][:self._pref_size], pos1)).replace(" ", "")
            ftr_str += " prev_pref&prev_pos=" + str((all_words[i-1][-self._suff_size:], pos1)).replace(" ", "")
        ftr_str += " curr_suf&prev_pos=" + str((all_words[i][:self._pref_size], pos1)).replace(" ", "")
        ftr_str += " curr_pref&prev_pos=" + str((all_words[i][-self._suff_size:], pos1)).replace(" ", "")
        if len(all_words) > i + 1:
            ftr_str += " next_suf&prev_pos=" + str((all_words[i+1][:self._pref_size], pos1)).replace(" ", "")
            ftr_str += " next_pref&prev_pos=" + str((all_words[i+1][-self._suff_size:], pos1)).replace(" ", "")
        return ftr_str

    def get_ftrs(self):
        self._update()
        return self._features_list

    @property
    def name(self):
        return "comb_word_pos"


class CostumeFtr(FeatureBuilder):
    def __init__(self):
        super(CostumeFtr, self).__init__(3)
        self._features_list = ["contain_hyphen=1", "contain_number=1", "contain_upper=1"]

    def learn_example(self, pos, example):
        pass

    def ftr_str(self, all_words, prev_pos, i):
        word = all_words[i]
        ftr_str = ""
        if any([c.isdigit() for c in word]):
            ftr_str += " contain_number=1"
        if any([c.isupper() for c in word]):
            ftr_str += " contain_upper=1"
        if any([c == "-" for c in word]):
            ftr_str += " contain_hyphen=1"
        return ftr_str

    @property
    def name(self):
        return "costume_"
