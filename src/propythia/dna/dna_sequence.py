"""
#########################################################################################

A class used for reading sequences.
The main objective is to create dataframes with valid sequences to calculate descriptors.

Authors: João Nuno Abreu 
Date: 03/2022
Email:

#########################################################################################
"""

from .utils import checker, checker_cut
import pandas as pd
import os
from Bio.SeqIO.FastaIO import SimpleFastaParser

class ReadDNA:
    def __init__(self):
        pass

    def read_sequence(self, sequence: str) -> str:
        """
        Reads a sequence, checks if it's valid and returns a dataframe with the sequence.
        """
        if checker(sequence):
            return sequence.strip().upper()
        else:
            raise ValueError("Error! Invalid character in sequence:", sequence)

    def read_fasta(self, filename: str, with_labels: bool = False) -> pd.DataFrame:
        """
        Reads the input file in fasta format. 
        It always reads sequences, and labels if the user wants.
        If the user wants the labels, he must specify the with_labels parameter as True and the FASTA format must be the following:
            >sequence_id1,label1
            ACTGACTG...
            >sequence_id2,label2
            ACTGACTG...
        """
        labels = []
        sequences = []
        
        if not os.path.isfile(filename):
            raise ValueError("Error! File does not exist:", filename)

        if 'fasta' not in filename:
            raise ValueError("Error! File must be in fasta format:", filename)
        
        with open(filename) as handle:
            for key, sequence in SimpleFastaParser(handle):
                # get label and check if it's valid
                if with_labels:
                    label = int(key.split(',')[1])
                    if(label not in [0,1]):
                        raise ValueError("Error! Label must be either 0 or 1 and it is:", label)
                    else:
                        labels.append(label)
                
                # get sequence and check if it's valid
                sequence = sequence.strip().upper()
                if checker(sequence):
                    sequences.append(sequence)      
                else:
                    raise ValueError("Error! Invalid character in sequence:", key)
        
        # add labels to result if the user wants
        if with_labels:
            return pd.DataFrame(list(zip(sequences, labels)), columns=['sequence', 'label'])
        else:
            return pd.DataFrame(sequences, columns=['sequence'])
            
    
    def read_csv(self, filename: str, with_labels: bool = False) -> pd.DataFrame:
        """
        Reads the input file in csv format. 
        It always reads sequences, and labels if the user wants. 
        There must be a column with the sequence.
        If the user wants the labels, he must specify the with_labels parameter as True and the column with the labels must be named "label".
        """
        
        if not os.path.isfile(filename):
            raise ValueError("Error! File does not exist:", filename)

        if 'csv' not in filename:
            raise ValueError("Error! File must be in csv format:", filename)
        
        dataset = pd.read_csv(filename)
        
        # check column names
        if 'sequence' not in dataset.columns:
            raise ValueError("The dataset must always have the column 'sequence'")
        if with_labels and 'label' not in dataset.columns:
            raise ValueError("Since with_labels is True, the dataset must have the column 'label'")
        
        # get sequences and labels
        sequences = dataset['sequence'].to_list()
        
        if with_labels:
            labels = dataset['label'].to_list()
        
        # check if sequences are valid
        valid_sequences = []
        for sequence in sequences:
            if checker(sequence) and "cut" not in filename:
                valid_sequences.append(sequence.strip().upper())  
            elif checker_cut(sequence) and "cut" in filename:
                valid_sequences.append(sequence.strip().upper())  
            else:
                raise ValueError("Error! Invalid character in sequence:", sequence)
            
        # check if labels are valid
        valid_labels = []
        if with_labels:
            for label in labels:
                if(label not in [0,1]):
                    raise ValueError("Error! Label must be either 0 or 1 and it is:", label)
                else:
                    valid_labels.append(label)
        
        # add labels to result if the user wants
        if with_labels:
            return pd.DataFrame(list(zip(sequences, labels)), columns=['sequence', 'label'])
        else:
            return pd.DataFrame(sequences, columns=['sequence'])
            
        
if __name__ == "__main__":
    reader = ReadDNA()
    data = reader.read_csv('datasets/primer/dataset.csv', with_labels=True)
    print(data)
    
    data = reader.read_fasta('datasets/primer/dataset.fasta', with_labels=True)
    print(data)
    
    