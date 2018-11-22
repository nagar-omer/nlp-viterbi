import sys
from MLETrain import *
from utils.viterbi import ViterbiAlg




if __name__ == "__main__":
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    out_file_name = sys.argv[4]
    extra_file_name = sys.argv[5]

    #emission_count, suffix_count = extract_e_from_file(e_mle_filename)
    #transition_count = extract_q_from_file(q_mle_filename)
    #transition_lambdas = extract_lambdas(extra_file_name)
    #emission, transition, suffix = calc_probabilities(transition_count, emission_count, suffix_count)

    me = MleExtractor(e_mle_filename, q_mle_filename, extra_file_name)
    #pos_list = list(set(list(me._transition_count[0].keys()) + [START]))


    src_file = open(input_file_name, "rt")  # open file
    tagger = ViterbiAlg(me._pos_list, me.prob_func)
    for line in src_file:
        # ---------- BREAK -----------
        seq = []
        label = []
        for w_p in line.split():  # break line to [.. (word, POS) ..]
            word, pos = w_p.rsplit("/", 1)
            seq.append(word)
            label.append(pos)
        pred = tagger.pred_viterbi(seq, log=True)  # predict

        # print results
        identical = sum([1 for p, l in zip(pred, label) if p == l])
        recall = str(int(identical / len(pred) * 100))
        print("pred: " + str(pred) + "\nlabel: " + str(label) +
              "\nrecall:\t" + str(identical) + "/" + str(len(pred)) + "\t~" + recall + "%")

