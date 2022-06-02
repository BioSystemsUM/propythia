# -*- coding: utf-8 -*-
"""
##############################################################################

A class used for reading sequences or change sequences.
The main objective is to create sequence objects to calculate descriptors
The class allows to:
     1) Read sequences from FASTA files
     2) Check if the DNA sequence is a valid sequence

Authors: João Nuno Abreu 

Date: 03/2022

Email:

##############################################################################
"""

from Bio.SeqIO.FastaIO import SimpleFastaParser
from utils import checker


class ReadDNA:
    """
    The ReadDNA class aims to read the input and transform it into a sequence that can be used to calculate Descriptors.
    """

    def __init__(self):
        self.d = {}

    def read_fasta(self, filename):
        """
        Reads the input file in fasta format. It reads all sequences. 
        """
        with open(filename) as handle:
            for key, sequence in SimpleFastaParser(handle):
                sequence = sequence.upper()
                if(checker(sequence)):
                    self.d[key] = sequence
                else:
                    print("Error! Invalid character in sequence.")
