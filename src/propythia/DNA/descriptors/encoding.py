"""
##############################################################################
A class  used for computing different types of DNA encoddings.
It contains encodings such one-hot-encoding. 
Authors: João Abreu
Date: 03/2022
Email:
##############################################################################
"""


class Encoding:

    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence.strip().upper()

    def get_binary(self):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates binary encoding. Each nucleotide is encoded by a four digit binary vector.
        :return: list with values of binary encoding
        """
        binary = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]
        }
        return [binary[i] for i in self.dna_sequence]
