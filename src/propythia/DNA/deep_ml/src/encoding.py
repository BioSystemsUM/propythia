"""
##############################################################################
A class  used for computing different types of DNA encodings.
It contains encodings such one-hot-encoding. 
Authors: João Abreu
Date: 03/2022
Email:
##############################################################################
"""

import sys
import numpy as np
sys.path.append('../')
from descriptors.utils import checker

class DNAEncoding:
    def __init__(self, dna_sequence: str):
        if(checker(dna_sequence)):
            self.dna_sequence = dna_sequence.strip().upper()

    def __init__(self, column: np.ndarray):
        arr = column.tolist()
        
        if(all(x == 1 or x == 0 for x in arr)):
            self.labels = column
            self.sequences = None
        elif(all(checker(x) for x in arr)):
            self.sequences = column
            self.labels = None
        else:
            print("Error! All labels must be either 0 or 1.")
            sys.exit(1)
    
    def one_hot_encode(self):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates binary encoding. Each nucleotide is encoded by a four digit binary vector.
        :return: list with values of binary encoding
        """
        binary = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            1: [1, 0],
            0: [0, 1]            
        }
        if(self.labels is not None):
            return np.array([binary[i] for i in self.labels])
        elif(self.sequences is not None):
            return np.array([[binary[i] for i in x] for x in self.sequences])
        else:
            print("Error! No labels or sequences were provided.")
            sys.exit(1)
