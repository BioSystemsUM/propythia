import numpy as np


def word_seq(seq, k, stride=1):
    i = 0
    words = []
    while i <= len(seq) - k:
        words.append(seq[i: i + k])
        i += stride
    return words


def create_dict(nucleotides):
    vec_dict = {}
    perms = k_len_perms(nucleotides, 3)
    perms.sort()
    for idx, seq in enumerate(perms):
        hot_vec = [0 for i in range(0, len(perms))]
        hot_vec[idx] = 1
        vec_dict[seq] = hot_vec
    return vec_dict


def k_len_perms(letters, k):
    n = len(letters)
    perms = []
    k_len_perms_hlpr(perms, letters, "", n, k)
    return perms


def k_len_perms_hlpr(perms, letters, prefix, n, k):
    if (k == 0):
        perms.append(prefix)
        return
    for i in range(0, n):
        newPrefix = prefix + letters[i]
        k_len_perms_hlpr(perms, letters, newPrefix, n, k - 1)


def create_rep_mat(words, hot_vec_dict, r_size):
    mat_len = len(words) - r_size + 1
    mat = [[] for i in range(0, mat_len)]
    i = 0
    while i < mat_len:
        j = i
        while j < i + r_size:
            mat[i].append(hot_vec_dict[words[j]])
            j += 1
        i += 1
    return mat


def get_rep_mat(seq, hot_vec_dict, k=3, r_size=2):
    words = word_seq(seq, k)
    rep_mat = create_rep_mat(words, hot_vec_dict, r_size)
    return rep_mat


def get_rep_mats(seqs):
    rep_mats = []
    hot_vec_dict = create_dict('ACGT')
    for seq in seqs:
        rep_mat = get_rep_mat(seq, hot_vec_dict, k=3, r_size=1)
        rep_mats.append(rep_mat)
    return rep_mats


def conv_labels(labels, dataset='splice'):
    converted = []
    for label in labels:
        if dataset == 'splice':
            if label == 'EI':
                converted.append(0)
            elif label == 'IE':
                converted.append(1)
            elif label == 'N':
                converted.append(2)
        elif dataset == 'promoter':
            if label == '+':
                converted.append(0)
            elif label == '-':
                converted.append(1)
    return converted


if __name__ == "__main__":
    # running on a test sequence matching example in paper
    seq = 'ACCGATTATGCA'
    words = word_seq(seq, 3)
    hot_vec_dict = create_dict('ACGT')
    rep_mat = create_rep_mat(words, hot_vec_dict, 2)


def to_categorical(y, num_classes=None, dtype='float32'):
    ''' 
    Source code of Keras' to_categorical() function. 
    '''
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
