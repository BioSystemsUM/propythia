import os
import pickle
import random
import sys
import math
import numpy as np
from itertools import product
import torch
import json
from torch import nn
from ray import tune

ALPHABET = 'ACGT'
ALPHABET_CUT = 'ACGTN'
pairs = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
}

combinations = {
    'mlp': ['descriptor'],
    'mlp_half': ['descriptor'],
    'cnn': ['one_hot', 'chemical', 'kmer_one_hot'],
    'cnn_half': ['one_hot', 'chemical', 'kmer_one_hot'],
    'lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
    'bi_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
    'gru': ['one_hot', 'chemical', 'kmer_one_hot'],
    'bi_gru': ['one_hot', 'chemical', 'kmer_one_hot'],
    'cnn_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
    'cnn_bi_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
    'cnn_gru': ['one_hot', 'chemical', 'kmer_one_hot'],
    'cnn_bi_gru': ['one_hot', 'chemical', 'kmer_one_hot']
}

def print_metrics(model_label, mode, data_dir, kmer_one_hot, class_weights, metrics):
    print("-" * 40)
    print("Results in test set: ")
    print("-" * 40)
    print("model:        ", model_label)
    print("mode:         ", mode)
    print("dataset:      ", data_dir.split("/")[-1])
    if mode == "kmer_one_hot":
        print("kmer_one_hot: ", kmer_one_hot)
    if "essential_genes" in data_dir:
        print("class_weights:", class_weights)
    print("-" * 40)

    for key in metrics:
        if key == 'confusion_matrix':
            print(f"{key:<{20}}= {metrics[key][0]}")
            print(f"{'':<{20}}  {metrics[key][1]}")
        else:
            print(f"{key:<{20}}= {metrics[key]:.3f}")

    print("-" * 40)

# -----------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def checker(sequence):
    """
    Checks if the input sequence is a valid DNA sequence.
    """
    return all(i in ALPHABET for i in sequence)

def checker_cut(sequence):
    """
    Checks if the input sequence is a valid DNA sequence. Includes the 'N' character as valid because it is used to fill the sequence to the right length.
    """
    return all(i in ALPHABET_CUT for i in sequence)

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
    nucleotides = [''.join(i) for i in product(ALPHABET_CUT, repeat=k)]
    encoded = []
    for i in range(5 ** k):
        encoded.append(np.zeros(5 ** k).tolist())
        encoded[i][i] = 1.0
        
    return {nucleotides[i]: encoded[i] for i in range(len(nucleotides))}

def calculate_kmer_list(sequence, k):
    l = []
    for i in range(len(sequence) - k + 1):
        l.append(sequence[i:i+k])
    return l

def read_config(device, filename='config.json'):
    """
    Reads the configuration file and validates the values. Returns the configuration.
    """
    with open(filename) as f:
        config = json.load(f)
    
    # ------------------------------------ check if data_dir exists ------------------------------------
    current_path = os.getcwd()
    current_path = current_path.replace("/notebooks", "") # when running from notebook
    
    config['combination']['data_dir'] = current_path + config['combination']['data_dir']
    if not os.path.exists(config['combination']['data_dir']):
        raise ValueError("Data directory does not exist:", config['combination']['data_dir'])    

    # --------------------------- check if model and mode combination is valid -------------------------
    model_label = config['combination']['model_label']
    mode = config['combination']['mode']
    if(model_label in combinations):
        if(mode not in combinations[model_label]):
            raise ValueError(model_label, 'does not support', mode, ', please choose one of', combinations[model_label])
    else:
        raise ValueError('Model label:', model_label, 'not implemented in', combinations.keys())

    # --------------------------- check if it's binary classification ----------------------------------
    loss = config['fixed_vals']['loss_function']
    output_size = config['fixed_vals']['output_size']
    if(loss != "cross_entropy" or output_size != 2):
        raise ValueError(
            'Model is not binary classification, please set loss_function to cross_entropy and output_size to 2')

    # --------------------------- create the cross entropy pytorch object ------------------------------
    class_weights = torch.tensor(config['combination']['class_weights']).to(device)
    config['fixed_vals']['loss_function'] = nn.CrossEntropyLoss(weight=class_weights)
    
    # --------------------------- create ray tune objects ----------------------------------------------
    config['hyperparameter_search_space']["hidden_size"] = tune.choice(config['hyperparameter_search_space']['hidden_size'])
    config['hyperparameter_search_space']["lr"] = tune.choice(config['hyperparameter_search_space']['lr'])
    config['hyperparameter_search_space']["batch_size"] = tune.choice(config['hyperparameter_search_space']['batch_size'])
    config['hyperparameter_search_space']["dropout"] = tune.choice(config['hyperparameter_search_space']['dropout'])
    
    if config['combination']['model_label'] not in ['mlp', 'cnn']:
        config['hyperparameter_search_space']["num_layers"] = tune.choice(config['hyperparameter_search_space']['num_layers'])
    
    return config

# ----------------------------------------------------------------------------------------------------
# ---------------------------------- Auxiliary function for models -----------------------------------
# ----------------------------------------------------------------------------------------------------

def calc_maxpool_output(hidden_size, sequence_length):
    conv1_padding = 0
    conv1_dilation = 1
    conv1_kernel_size = 12
    conv1_stride = 1

    l_out = ((sequence_length + 2*conv1_padding - conv1_dilation*(conv1_kernel_size-1) - 1)/conv1_stride + 1)
    maxpool_padding = 0
    maxpool_dilation = 1
    maxpool_stride = 5
    maxpool_kernel_size = 12
    max_pool_output = int((l_out+2*maxpool_padding-maxpool_dilation*(maxpool_kernel_size-1)-1)/maxpool_stride+1)

    max_pool_output *= hidden_size
    
    return max_pool_output


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
