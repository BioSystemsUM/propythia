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

import glob
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
        d = {}
        with open(filename) as handle:
            for values in SimpleFastaParser(handle):
                if(checker(values[1].upper())):
                    d[values[0]] = values[1].upper()
                else:
                    print("Error! Invalid character in sequence.")
        return d

    def read_fasta_in_folder(self, folder):
        """
        Reads all files in a folder in fasta format. It reads all sequences. 
        """
        for filename in glob.glob(folder + '/*.fasta'):
            self.d[filename[len(folder)+1:-6]] = self.read_fasta(filename)
