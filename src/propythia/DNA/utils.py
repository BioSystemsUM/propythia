import os
import pickle
import sys
import math
import numpy as np
from itertools import product

ALPHABET = 'ACGT'
pairs = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
}

# -----------------------------------------------------------------------------


def checker(sequence):
    """
    Checks if the input sequence is a valid DNA sequence.
    """
    return all(i in ALPHABET for i in sequence)

def normal_round(n):
    """
    Equivalent to python's round but its rounds up if it ends up with 0.5.
    """
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def normalize_dict(dic):
    """Normalize the value of a dictionary."""
    N = sum(dic.values())
    for key in dic:
        dic[key] = round(dic[key] / N, 3)
    return dic


def make_kmer_list(k):
    try:
        return ["".join(e) for e in product(ALPHABET, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def make_kmer_dict(k):
    try:
        return {''.join(i): 0 for i in product(ALPHABET, repeat=k)}
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError

def calculate_kmer_onehot(k):
    nucleotides = [''.join(i) for i in product(ALPHABET, repeat=k)]
    encoded = []
    for i in range(4 ** k):
        encoded.append(np.zeros(4 ** k).tolist())
        encoded[i][i] = 1.0
        
    return {nucleotides[i]: encoded[i] for i in range(len(nucleotides))}

def calculate_kmer_list(sequence, k):
    l = []
    for i in range(len(sequence) - k + 1):
        l.append(sequence[i:i+k])
    return l

# ----------------------------------------------------------------------------------------------------
# ------------------- The following functions were retrieved from repDNA package ---------------------
# ----------------------------------------------------------------------------------------------------

def ready_acc(k, phyche_index=None, all_property=False, extra_phyche_index=None):
    """Public function for get sequence_list and phyche_value.
    """
    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}
    phyche_value = generate_phyche_value(k, phyche_index, all_property, extra_phyche_index)

    return phyche_value


def generate_phyche_value(k, phyche_index=None, all_property=False, extra_phyche_index=None):
    """Combine the user selected phyche_list, is_all_property and extra_phyche_index to a new standard phyche_value."""
    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}

    diphyche_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                     'A-philicity', 'Propeller twist', 'Duplex stability:(freeenergy)',
                     'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                     'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                     'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                     'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                     'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                     'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt',
                     'Roll', 'Shift', 'Slide', 'Rise']
    triphyche_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                      'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                      'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']

    # Set and check physicochemical properties.
    if 2 == k:
        if all_property is True:
            phyche_index = diphyche_list
        else:
            for e in phyche_index:
                if e not in diphyche_list:
                    raise ValueError(" ".join(["Sorry, the physicochemical properties", e, "is not exit."]))
    elif 3 == k:
        if all_property is True:
            phyche_index = triphyche_list
        else:
            for e in phyche_index:
                if e not in triphyche_list:
                    raise ValueError(" ".join(["Sorry, the physicochemical properties", e, "is not exit."]))

    return extend_phyche_index(get_phyche_index(k, phyche_index), extra_phyche_index)


def get_phyche_index(k, phyche_list):
    """get phyche_value according phyche_list."""
    phyche_value = {}
    if 0 == len(phyche_list):
        for nucleotide in make_kmer_list(k):
            phyche_value[nucleotide] = []
        return phyche_value

    nucleotide_phyche_value = get_phyche_factor_dic(k)
    for nucleotide in make_kmer_list(k):
        if nucleotide not in phyche_value:
            phyche_value[nucleotide] = []
        for e in nucleotide_phyche_value[nucleotide]:
            if e[0] in phyche_list:
                phyche_value[nucleotide].append(e[1])

    return phyche_value


def extend_phyche_index(original_index, extend_index):
    """Extend {phyche:[value, ... ]}"""
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index


def get_phyche_factor_dic(k):
    """Get all {nucleotide: [(phyche, value), ...]} dict."""
    full_path = os.path.realpath(__file__)
    if 2 == k:
        file_path = "%s/data/mmc3.data" % os.path.dirname(full_path)
    elif 3 == k:
        file_path = "%s/data/mmc4.data" % os.path.dirname(full_path)
    else:
        sys.stderr.write("The k can just be 2 or 3.")
        sys.exit(0)

    try:
        with open(file_path, 'rb') as f:
            phyche_factor_dic = pickle.load(f)
    except:
        with open(file_path, 'r') as f:
            phyche_factor_dic = pickle.load(f)

    return phyche_factor_dic


def make_ac_vector(sequence_list, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    vec_ac = []
    for sequence in sequence_list:
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for j in range(len_phyche_value):

                # Calculate average phyche_value for a nucleotide.
                ave_phyche_value = 0.0
                for i in range(len_seq - temp_lag - k + 1):
                    nucleotide = sequence[i: i + k]
                    ave_phyche_value += float(phyche_value[nucleotide][j])
                ave_phyche_value /= len_seq

                # Calculate the vector.
                temp_sum = 0.0
                for i in range(len_seq - temp_lag - k + 1):
                    nucleotide1 = sequence[i: i + k]
                    nucleotide2 = sequence[i + temp_lag: i + temp_lag + k]
                    temp_sum += (float(phyche_value[nucleotide1][j]) - ave_phyche_value) * (
                        float(phyche_value[nucleotide2][j]))

                try:
                    val = round(temp_sum / (len_seq - temp_lag - k + 1), 3)
                except ZeroDivisionError:
                    val = 0.0
                each_vec.append(val)
        vec_ac.append(each_vec)

    return vec_ac


def make_cc_vector(sequence_list, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    vec_cc = []
    for sequence in sequence_list:
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for i1 in range(len_phyche_value):
                for i2 in range(len_phyche_value):
                    if i1 != i2:
                        # Calculate average phyche_value for a nucleotide.
                        ave_phyche_value1 = 0.0
                        ave_phyche_value2 = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide = sequence[j: j + k]
                            ave_phyche_value1 += float(phyche_value[nucleotide][i1])
                            ave_phyche_value2 += float(phyche_value[nucleotide][i2])
                        ave_phyche_value1 /= len_seq
                        ave_phyche_value2 /= len_seq

                        # Calculate the vector.
                        temp_sum = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide1 = sequence[j: j + k]
                            nucleotide2 = sequence[j + temp_lag: j + temp_lag + k]
                            temp_sum += (float(phyche_value[nucleotide1][i1]) - ave_phyche_value1) * \
                                        (float(phyche_value[nucleotide2][i2]) - ave_phyche_value2)
                        try:
                            val = round(temp_sum / (len_seq - temp_lag - k + 1), 3)
                        except ZeroDivisionError:
                            val = 0.0
                        each_vec.append(val)

        vec_cc.append(each_vec)

    return vec_cc
