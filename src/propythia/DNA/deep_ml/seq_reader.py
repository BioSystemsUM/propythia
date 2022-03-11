from one_hot_rep import get_rep_mats, conv_labels


def load_data(fname):
    seqs = []
    labels = []
    f = open(fname)
    for line in f:
        line_no_wspace = line.replace(" ", "")
        line_no_nwline = line_no_wspace.replace("\n", "")
        line_arr = line_no_nwline.split(",")
        label = line_arr[0]
        seq = line_arr[2]
        # sequence cleaning
        seq = seq.upper()    # b/c rep matrix built on uppercase
        seq = seq.replace("\t", "")      # present in promoter
        seq = seq.replace("N", "A")  # undetermined nucleotides in splice
        seq = seq.replace("D", "G")
        seq = seq.replace("S", "C")
        seq = seq.replace("R", "G")

        labels.append(label)
        seqs.append(seq)
    f.close()
    return seqs, labels
