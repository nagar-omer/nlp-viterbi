import sys
from MLETrain import *
START = "START"


def pred_greedy(me, seq, log=True):
    poses = []
    t1 = START
    t2 = None
    for word in seq:
        max_pos = None
        max_prob = -10000000
        l = [pos for pos in me._pos_list if pos != START]
        for pos in l:
            em = emission((word, pos), me._emission, me._suffix, log)
            if t2 is not None:
                tran = transition((pos, t1, t2), me._transition, me._transition_lambdas, log)
            else:
                tran = transition((pos, t1), me._transition, me._transition_lambdas, log)
            if log:
                prob = em + tran
            else:
                prob = em * tran
            if prob > max_prob:
                max_prob = prob
                max_pos = pos
        t2 = t1
        t1 = max_pos
        poses.append(max_pos)
    return poses




if __name__ == "__main__":
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    out_file_name = sys.argv[4]
    extra_file_name = sys.argv[5]

    me = MleExtractor(e_mle_filename, q_mle_filename, extra_file_name)

    src_file = open(input_file_name, "rt")
    out_file = open(out_file_name, "wt")
    for line in src_file:
        seq = []
        label = []
        for w_p in line.split():  # break line to [.. (word, POS) ..]
            word, pos = w_p.rsplit("/", 1)
            seq.append(word)
            label.append(pos)
        pred = pred_greedy(me, seq, log=True)
        for i, word in enumerate(seq):
            out_file.write(word + "/" + str(pred[i]) + " ")
        out_file.write("\n")

        # print results
        identical = sum([1 for p, l in zip(pred, label) if p == l])
        recall = str(int(identical / len(pred) * 100))
        print("pred: " + str(pred) + "\nlabel: " + str(label) +
              "\nrecall:\t" + str(identical) + "/" + str(len(pred)) + "\t~" + recall + "%")




