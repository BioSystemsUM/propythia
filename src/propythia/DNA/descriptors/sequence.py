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
        pass
        
    def read_fasta(self, filename):
        """
        Reads the input file in fasta format. It reads all sequences. 
        """
        self.d = {}
        with open(filename) as handle:
            for values in SimpleFastaParser(handle):
                if(checker(values[1])):
                    self.d[values[0]] = values[1]
                else:
                    print("Error! Invalid character in sequence.")
        return self.d

    def read_one_sequence_fasta(self, filename):
        """
        Reads the input file in fasta format. It only reads one sequence.
        """
        with open(filename) as handle:
            for index, values in enumerate(SimpleFastaParser(handle)):
                if(index != 0):
                    print("Error! Please input only one sequence.")
                    break
                
                if(checker(values[1])):
                        self.dna_sequence = values[1]
                else:
                    print("Error! Invalid character in sequence.")
        return self.dna_sequence
