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
import pandas as pd
import numpy as np
sys.path.append('../')
from descriptors.utils import checker

class DNAEncoding:
    def __init__(self, dna_sequence: str):
        if(checker(dna_sequence)):
            self.dna_sequence = dna_sequence.strip().upper()

    def __init__(self, df: pd.DataFrame):
        
        if(df.columns.tolist() != ['sequence', 'label']):
            print("Error! The dataframe must have two columns only: sequence and label.")
            sys.exit(1)
        
        for i in df['sequence']:
            if(checker(i) == False):
                print("Error! Invalid character in sequence.")
                sys.exit(1)
                
        self.df = df

    def one_hot_encode(self, enconde_target: bool = False):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates binary encoding. Each nucleotide is encoded by a four digit binary vector.
        :param enconde_target: If True, the target column will be encoded.
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

        if(self.df is not None):
            self.df['sequence'] = self.df['sequence'].apply(lambda x: np.array([binary[i] for i in x]))
            if(enconde_target):
                self.df['label'] = self.df['label'].apply(lambda x: np.array(binary[x]))
            return self.df
        else:
            return [binary[i] for i in self.dna_sequence]
        
