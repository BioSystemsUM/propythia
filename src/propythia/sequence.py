# -*- coding: utf-8 -*-
"""
##############################################################################

A class used for reading sequences or change sequences.
The main objective is to create sequence objects to calculate descriptors
The class allows to:
     1)Read sequences from string or from uniprot ID (is also possible retrieve sequences fom txt with uniprot IDs)
     2)Check if the protein sequence is a valid sequence
     3)Obtain a sized sequence of list of sequences, adding or cutting from both n and c terminals
     4)From one sequence generate list of subsequences based on sliding window approach, from specific aa, from the
     the terminals or divide the sequence in parts

Authors:Ana Marta Sequeira

Date: 01/2019

Email:

##############################################################################
"""

from propythia.adjuv_functions.sequence.get_sequence import get_protein_sequence, get_protein_sequence_from_txt
from propythia.adjuv_functions.sequence.pro_check import protein_check
from propythia.adjuv_functions.sequence.get_sized_seq import seq_equal_lenght
from propythia.adjuv_functions.sequence.get_sub_seq import sub_seq_sliding_window, sub_seq_to_aa, sub_seq_split,sub_seq_terminals


class ReadSequence:

    """
    The ReadSequence class aims to read the input and transform it into a sequence that can be used to calculate Descriptors.
    It accepts uniprotID, txt with uniprot ID and string with aminoacid sequence. The functions to read protein are based on the package pydpi.
    It allows to get sized sequences or a variety of subsequences from just one sequence.
    """

    def __init__(self):
        """	constructor """
        pass

# #########################################
    # read string or ID and get protein sequence

    def read_protein_sequence(self, protein_sequence=""):
        """
        Read a protein sequence.
        :param protein_sequence: String with sequence
        :return: String with aa sequence
        """

        protein_sequence=str.strip(protein_sequence)
        index=protein_check(protein_sequence)

        if index==0:
            print("Error......")
            print("Please input a correct protein.")
        else:
            self.protein_sequence=protein_sequence
            return self.protein_sequence

    def get_protein_sequence_from_txt(self, path, openfile, savefile):
        """
        Function to retrieve sequences from a txt with uniprot ID. Does not retrieve a sequence object
        :param path: directory path containing ID file
        :param openfile:ID file ('name.txt')
        :param savefile: saved file with obtained protein sequences ('name2.txt')
        :return:File containing string of sequences
        """
        self.txt=get_protein_sequence_from_txt(path,openfile,savefile)
        return self.txt

    def get_protein_sequence_from_id(self, uniprotid=""):
        """
        Downloading a protein sequence by uniprot id.
        :param uniprotid: String with UniprotID
        :return: String with aa sequence
        """
        self.protein=get_protein_sequence(str.strip(uniprotid))
        index=protein_check(self.protein)

        if index==0:
            print("Error......")
            print("Please input a correct protein.")
        else:
            self.protein_sequence=self.protein
        return self.protein_sequence

##########################################
    # check protein

    def checkprotein(self, protein_sequence=""):
        """
        Check whether the protein sequence is a valid amino acid sequence or not.
        Just check. Not assign to the object.
        :param protein_sequence: protein sequence
        :return: error or valid message.
        """

        index=protein_check(protein_sequence)

        if index==0:
            print("Error......")
            print("Please input a correct protein.")
        else:
            print('sequence valid')

##########################################
    # Get equal size sequences

    def get_sized_seq(self, sequences=[], n_terminal=10, c_terminal=10, terminal=0):
        """
        cut or add aminoacids to obtain sequences with equal lenght.
        :param sequences: list containing protein sequences (string) or just a protein sequence
        :param n_terminal: number of aa to consider in the n terminal (left side of sequence)
        :param c_terminal: number of aa to consider in the c terminal (right side of sequence)
        :param terminal: in case of the need to add dummy aa and no terminal as already been chosen, it decides where to add
                        0 to add to the right (consider N terminal)
                        -1 to add to the left (consider C terminal)
                        2 to add in the middle (N and C terminal will be both present and repeated with dummy in middle
        :return: list of sequences containing all the same lenght. if just one sequence given it will return a string
        """
        if len(sequences)==1:
            seq=seq_equal_lenght(sequences[0],n_terminal,c_terminal,terminal)
            return seq
        else:
            equal_size_sequences=[] #list to store sequences with equal lenght
            for seq in sequences:
                seq_2=seq_equal_lenght(seq,n_terminal,c_terminal,terminal)
                equal_size_sequences.append(seq_2)
            return equal_size_sequences

##########################################
    # Generate subsequences

    def get_sub_seq_sliding_window(self, seq, window_size=20, gap=1, index=True):
        """
        sliding window of the protein given. It will generate a list of n sequences with
        lenght equal to the value of window and spaced a gap value. It can or not retrieve
        the indices of location of the subsequence in the original sequence.

        :param seq: protein sequence
        :param window_size: number of aminoacids to considerer, lenght of the subsequence. for default 20
        :param gap: gap size of the search of windows in sequence. default 1
        :param index: if true, return the indices of location of the subsequence in the original sequence
        :return: list with subsequences generated with or without a list of tuples with location of subsequences
         in original sequence
        """

        list_sliding_window=sub_seq_sliding_window(seq,window_size,gap,index)
        return list_sliding_window

    def get_sub_seq_to_aa(self, seq, ToAA='S', window=5):
        """
        Get all 2*window+1 sub-sequences whose center is ToAA in a protein

        :param seq:  protein sequence
        :param ToAA: central (query point) amino acid in the sub-sequence
        :param window:  span (number of amnoacids to go front and back from the ToAA
        :return: list form containing sub-sequences
        """

        list_subseq_to_aa=sub_seq_to_aa(seq, ToAA, window)
        return list_subseq_to_aa

    def get_sub_seq_split(self, seq, number_of_subseq):
        """
        Split the originl seq in n number of subsequences.
        :param seq: protein sequence
        :param number_of_subseq: number of subsequences to divide the original seq
        :return: list with number_of_sequences sequences.

        """
        list_sub_seq_split=sub_seq_split(seq,number_of_subseq)
        return list_sub_seq_split

    def get_sub_seq_terminals(self, seq, N_terminal=5, C_terminal=5, rest=True):
        """
        Divide the sequence in the N terminal and  C terminal with sizes defined by the user. It returns
        a list with N and C terminal and rest of the sequence.
        By default the N terminal is considerer to be the beggining of the sequence (left)

        :param seq: protein sequence
        :param N_terminal: size of the N terminal to consider. If zero will not return
        :param C_terminal: size of the C terminal to consider. If zero will not return
        :param rest: If true will return the restant subsequence
        :return: list with N, C and rest of the sequence
        """
        list_sub_seq_term= sub_seq_terminals(seq, N_terminal, C_terminal,rest)
        return list_sub_seq_term

