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
from typing import Union
import numpy as np
sys.path.append('../')
from .utils import calculate_kmer_onehot, calculate_kmer_list

class DNAEncoder:
    def __init__(self, data: Union[str, np.ndarray]):
        
        if(isinstance(data, str)):
            self.dna_sequence = data.strip().upper()
            self.sequences = None
        else:
            self.sequences = data
            self.dna_sequence = None
              
    def one_hot_encode(self, dimension = 3):
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
            'N': [0, 0, 0, 0],
            1: [1, 0],
            0: [0, 1]            
        }
        
        binary2 = {
            'A': 1,
            'C': 2,
            'G': 3,
            'T': 4,
            'N': 0,
        }
        
        values = binary if dimension == 3 else binary2
        
        if(self.sequences is not None):
            return np.array([[values[i] for i in x] for x in self.sequences])
        elif(self.dna_sequence is not None):
            return np.array([values[i] for i in self.dna_sequence])
        else:
            print("Unexpected error: self.sequences and self.dna_sequence are None.")
            sys.exit(1)
            
    def chemical_encode(self):
        """
        From: https://academic.oup.com/bioinformatics/article/33/22/3518/4036387

        Calculates nucleotide chemical property

        Chemical property | Class	   | Nucleotides
        -------------------------------------------
        Ring structure 	  | Purine 	   | A, G
                          | Pyrimidine | C, T
        -------------------------------------------
        Hydrogen bond     | Weak 	   | A, T
         	              | Strong 	   | C, G
        -------------------------------------------
        Functional group  | Amino 	   | A, C
                          | Keto 	   | G, T

        :return: list with values of nucleotide chemical property
        """
        chemical_property = {
            'A': [1, 1, 1],
            'C': [0, 0, 1],
            'G': [1, 0, 0],
            'T': [0, 1, 0],
            'N': [0, 0, 0],
            1: [1, 0],
            0: [0, 1]
        }
        if(self.sequences is not None):
            return np.array([[chemical_property[i] for i in x] for x in self.sequences])
        elif(self.dna_sequence is not None):
            return np.array([chemical_property[i] for i in self.dna_sequence])
        else:
            print("Unexpected error: self.sequences and self.dna_sequence are None.")
            sys.exit(1)

    def kmer_one_hot_encode(self, k):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates binary encoding. Each nucleotide is encoded by a four digit binary vector.
        :return: list with values of binary encoding
        """
        if(self.sequences is not None):
            res = []
            d = calculate_kmer_onehot(k)
            for sequence in self.sequences:
                l = calculate_kmer_list(sequence, k)
                res.append(np.array([d[i] for i in l]))
            return np.array(res)
        elif(self.dna_sequence is not None):
            d = calculate_kmer_onehot(k)
            l = calculate_kmer_list(self.dna_sequence, k)
            return np.array([d[i] for i in l])
        else:
            print("Unexpected error: self.sequences and self.dna_sequence are None.")
            sys.exit(1)

if __name__ == "__main__":
    encoder = DNAEncoder("ACGT")
    print(encoder.one_hot_encode())
    print(encoder.chemical_encode())
    print(encoder.kmer_one_hot_encode(2))