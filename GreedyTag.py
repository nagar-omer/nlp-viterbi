import sys


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
    for line in src_file:
        format, count = line.split("\t")
        word, pos = format.split()
        emission_count[(word, pos)] = int(count)
        return emission_count


def extract_q_from_file(q_file):
    src_file = open(q_file, "rt")  # open file
    transition_count = {0: {}, 1: {}, 2: {}}
    for line in src_file:
        pos_seq, count = line.split("\t")
        pos_seq = tuple(pos_seq.split())
        transition_count[len(pos_seq) - 1][pos_seq] = int(count)
    return transition_count

def extract_lambdas(file):
    src_file = open(file, "rt")  # open file
    transition_lambdas = []
    for line in src_file:
        transition_lambdas.append(int(line))
    return transition_lambdas

if __name__ == "__main__":
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    out_file_name = sys.argv[4]
    extra_file_name = sys.argv[5]

