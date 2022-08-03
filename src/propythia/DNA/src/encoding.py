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
from utils import checker, calculate_kmer_onehot, calculate_kmer_list

class DNAEncoding:
    
    def __init__(self, *args):
        if isinstance(args[0], str):
            dna_sequence = args[0]
            if(checker(dna_sequence)):
                self.dna_sequence = dna_sequence.strip().upper()
                self.labels = None
                self.sequences = None
            else:
                print("Error! Letters of DNA don't belong to [A,C,T,G]:", dna_sequence)
                sys.exit(1)
        elif isinstance(args[0], np.ndarray):
            column = args[0]
            arr = column.tolist()
            if(all(x in [0,1] for x in arr)):
                self.labels = column
                self.sequences = None
            elif(all(checker(x) for x in arr)):
                self.sequences = column
                self.labels = None
            else:
                print("Error! All labels must be either 0 or 1.")
                sys.exit(1)
        else:
            print("Error! Invalid input type:", type(args[0]), "Expected: str or np.ndarray")
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
        elif(self.dna_sequence is not None):
            return np.array([binary[i] for i in self.dna_sequence])
        else:
            print("Error! No labels or sequences were provided.")
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
        Hydrogen bond 	  | Strong 	   | C, G
                          | Weak 	   | A, T
        -------------------------------------------
        Functional group  | Amino 	   | A, C
                          | Keto 	   | G, T

        :return: list with values of nucleotide chemical property
        """
        chemical_property = {
            'A': [1, 1, 1],
            'C': [0, 1, 0],
            'G': [1, 0, 0],
            'T': [0, 0, 1],
            1: [1, 0],
            0: [0, 1]
        }
        if(self.labels is not None):
            return np.array([chemical_property[i] for i in self.labels])
        elif(self.sequences is not None):
            return np.array([[chemical_property[i] for i in x] for x in self.sequences])
        elif(self.dna_sequence is not None):
            return np.array([chemical_property[i] for i in self.dna_sequence])
        else:
            print("Error! No labels or sequences were provided.")
            sys.exit(1)

    def kmer_one_hot_encode(self, k):
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
            res = []
            for sequence in self.sequences:
                d = calculate_kmer_onehot(k)
                l = calculate_kmer_list(sequence, k)
                res.append(np.array([d[i] for i in l]))
            return np.array(res)
        elif(self.dna_sequence is not None):
            d = calculate_kmer_onehot(k)
            l = calculate_kmer_list(self.dna_sequence, k)
            return np.array([d[i] for i in l])
        else:
            print("Error! No labels or sequences were provided.")
            sys.exit(1)

if __name__ == "__main__":
    encoder = DNAEncoding("ATGC")
    print(encoder.one_hot_encode())