"""
#########################################################################################

A class used for reading sequences.
The main objective is to create dataframes with valid sequences to calculate descriptors.

Authors: João Nuno Abreu 
Date: 03/2022
Email:

#########################################################################################
"""

import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from utils import checker


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
        dataset = pd.read_csv(filename)
        
        # check column names
        if 'sequence' not in dataset.columns:
            raise ValueError("The dataset must always have the column 'sequence'")
        if with_labels and 'label' not in dataset.columns:
            raise ValueError("Since with_labels is True, the dataset must have the column 'label'")
        
        # get sequences and labels
        sequences = dataset['sequence'].to_list()
        labels = dataset['label'].to_list()
        
        # check if sequences are valid
        count = 0
        valid_sequences = []
        for sequence in sequences:
            if checker(sequence):
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
    data = reader.read_csv('data/dataset.csv', with_labels=True)
    print(data)
    
    data = reader.read_fasta('data/dataset.fasta', with_labels=True)
    print(data)
    