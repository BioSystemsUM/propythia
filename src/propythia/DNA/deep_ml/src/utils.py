import numpy as np
from itertools import product
ALPHABET = 'ACGT'

def checker(sequence):
    """
    Checks if the input sequence is a valid DNA sequence.
    """
    return all(i in ALPHABET for i in sequence)

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