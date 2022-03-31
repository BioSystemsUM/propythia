# -*- coding: utf-8 -*-
"""
##############################################################################

This function gives binary profile of aminoacid composition

It receives a sequence and returns a dictionary containing the binary profile


Authors: Ana Marta Sequeira

Date: 05/ 2019

Email:

##############################################################################
"""
# import pandas as pd
# import sys
# import os
# import numpy as np
# import getopt
from tensorflow.keras.utils import to_categorical


def bin_aa_ct(seq, alphabet = "ARNDCEQGHILKMFPSTWYV"):
    """
    Transform sequency in binary profile. If the alphabet provided not include the B, Z, U and O but these aminoacids
    are present in sequence they will be substitute for the closest aa ( N, Q, C and K respectively). If X is presented
    in sequence but not in the alphabet given, it will be eliminated.
    :param alphabet: alphabet to consider. by defult a 20 aa alphabet "ARNDCEQGHILKMFPSTWYV"
    :param seq: protein sequence
    :return: binary matrix of sequence
    """

    if len(alphabet) < 25:  # alphabet x or alphabet normal
        seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
        seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
        seq3 = seq2.replace('U',
                            'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
        seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
        seq = seq4
        if len(alphabet) == 20:  # alphabet normal substitute every letters
            seq = seq4.replace('X', '')  # unknown character eliminated

    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in seq]
    encoded = to_categorical(integer_encoded)
    return encoded


    #
    # #print("B_C0,B_C0_1,B_C0_2,B_C0_3,B_C0_4,B_C0_5,B_C0_6,B_C0_7,B_C0_8,B_C0_9,B_C0_10,B_C0_11,B_C0_12,B_C0_13,B_C0_14,B_C0_15,B_C0_16,B_C0_17,B_C0_18,B_C0_19,B_C1,B_C1_1,B_C1_2,B_C1_3,B_C1_4,B_C1_5,B_C1_6,B_C1_7,B_C1_8,B_C1_9,B_C1_10,B_C1_11,B_C1_12,B_C1_13,B_C1_14,B_C1_15,B_C1_16,B_C1_17,B_C1_18,B_C1_19,B_C2,B_C2_1,B_C2_2,B_C2_3,B_C2_4,B_C2_5,B_C2_6,B_C2_7,B_C2_8,B_C2_9,B_C2_10,B_C2_11,B_C2_12,B_C2_13,B_C2_14,B_C2_15,B_C2_16,B_C2_17,B_C2_18,B_C2_19,B_C3,B_C3_1,B_C3_2,B_C3_3,B_C3_4,B_C3_5,B_C3_6,B_C3_7,B_C3_8,B_C3_9,B_C3_10,B_C3_11,B_C3_12,B_C3_13,B_C3_14,B_C3_15,B_C3_16,B_C3_17,B_C3_18,B_C3_19,B_C4,B_C4_1,B_C4_2,B_C4_3,B_C4_4,B_C4_5,B_C4_6,B_C4_7,B_C4_8,B_C4_9,B_C4_10,B_C4_11,B_C4_12,B_C4_13,B_C4_14,B_C4_15,B_C4_16,B_C4_17,B_C4_18,B_C4_19,B_C5,B_C5_1,B_C5_2,B_C5_3,B_C5_4,B_C5_5,B_C5_6,B_C5_7,B_C5_8,B_C5_9,B_C5_10,B_C5_11,B_C5_12,B_C5_13,B_C5_14,B_C5_15,B_C5_16,B_C5_17,B_C5_18,B_C5_19,B_C6,B_C6_1,B_C6_2,B_C6_3,B_C6_4,B_C6_5,B_C6_6,B_C6_7,B_C6_8,B_C6_9,B_C6_10,B_C6_11,B_C6_12,B_C6_13,B_C6_14,B_C6_15,B_C6_16,B_C6_17,B_C6_18,B_C6_19,B_C7,B_C7_1,B_C7_2,B_C7_3,B_C7_4,B_C7_5,B_C7_6,B_C7_7,B_C7_8,B_C7_9,B_C7_10,B_C7_11,B_C7_12,B_C7_13,B_C7_14,B_C7_15,B_C7_16,B_C7_17,B_C7_18,B_C7_19,B_C8,B_C8_1,B_C8_2,B_C8_3,B_C8_4,B_C8_5,B_C8_6,B_C8_7,B_C8_8,B_C8_9,B_C8_10,B_C8_11,B_C8_12,B_C8_13,B_C8_14,B_C8_15,B_C8_16,B_C8_17,B_C8_18,B_C8_19,B_C9,B_C9_1,B_C9_2,B_C9_3,B_C9_4,B_C9_5,B_C9_6,B_C9_7,B_C9_8,B_C9_9,B_C9_10,B_C9_11,B_C9_12,B_C9_13,B_C9_14,B_C9_15,B_C9_16,B_C9_17,B_C9_18,B_C9_19,")
    #
    # A=('1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # C=('0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # D=('0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # E=('0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # F=('0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # G=('0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # H=('0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # I=('0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0')
    # K=('0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0')
    # L=('0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0')
    # M=('0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0')
    # N=('0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0')
    # P=('0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0')
    # Q=('0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0')
    # R=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0')
    # S=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0')
    # T=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0')
    # V=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0')
    # W=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0')
    # Y=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1')
    # Z=('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    # matrix=[]
    # for i in seq: i.upper()
    # for i in range(0,len(zz)):
    #     for j in zz[i]:
    #         if j == "A":
    #            matrix.append(A)
    #         if j == "C":
    #             matrix.append(C)
    #         if j == "D":
    #             matrix.append(D)
    #         if j == "E":
    #             matrix.append(E)
    #         if j == "F":
    #             matrix.append(F)
    #         if j == "G":
    #             matrix.append(G)
    #         if j == "H":
    #             matrix.append(H)
    #         if j == "I":
    #             matrix.append(I)
    #         if j == "K":
    #             matrix.append(K)
    #         if j == "L":
    #             matrix.append(L)
    #         if j == "M":
    #             matrix.append(M)
    #         if j == "N":
    #             matrix.append(N)
    #         if j == "P":
    #             matrix.append(P)
    #         if j == "Q":
    #             matrix.append(Q)
    #         if j == "R":
    #             matrix.append(R)
    #         if j == "S":
    #             matrix.append(S)
    #         if j == "T":
    #             matrix.append(T)
    #         if j == "V":
    #             matrix.append(V)
    #         if j == "W":
    #             matrix.append(W)
    #         if j == "Y":
    #             matrix.append(Y)
    #         if j == "Z":
    #             matrix.append(Z) #DUMMY AA




if __name__ == "__main__":
    print(bin_aa_ct('AYA', alphabet = "ARNDCEQGHILKMFPSTWYV"))
    print(bin_aa_ct('AYAACPPPQSTY', alphabet = "ARNDCEQGHILKMFPSTWYV"))

