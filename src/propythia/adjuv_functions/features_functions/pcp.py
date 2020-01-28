# coding: utf-8

#!/usr/bin/env python

"""

Function to calculate fraction of each physico-chemical property present in sequences (separated by comma)
It receives a sequence and a list containing feature numbers (valid number, 0-24)

    FEATURE NAME                    FEATURE NUMBER

    'Positively charged' --                  0
    'Negatively charged' --                  1
    'Neutral charged' --                     2
    'Polarity' --                            3
    'Non polarity' --                        4
    'Aliphaticity' --                        5
    'Cyclic' --                              6
    'Aromaticity' --                         7
    'Acidicity'--                            8
    'Basicity'--                             9
    'Neutral (ph)' --                       10
    'Hydrophobicity' --                     11
    'Hydrophilicity' --                     12
    'Neutral' --                            13
    'Hydroxylic' --                         14
    'Sulphur content' -                     15
    'Secondary Structure(Helix)'            16
    'Secondary Structure(Strands)',         17
    'Secondary Structure(Coil)',            18
    'Solvent Accessibilty (Buried)',        19
    'Solvent Accesibilty(Exposed)',         20
    'Solvent Accesibilty(Intermediate)',    21
    'Tiny',                                 22
    'Small',                                23
    'Large'                                 24
Author: Ana Marta Sequeira

email:

date: 05/2019
"""



# Secondary functions, move to next section

#Single function to calculate 30 physico che
import sys
import pandas as pd
import numpy as np
import csv
import getopt

#Finding physico-chemical property of a vector of polypeptides

PCP= pd.read_csv('data/PhysicoChemical.csv', header=None) #Our reference table for properties

headers = ['Positively charged',
           'Negatively charged',
           'Neutral charged',
           'Polarity',
           'Non polarity',
           'Aliphaticity',
           'Cyclic',
           'Aromaticity',
           'Acidicity',
           'Basicity',
           'Neutral (ph)',
           'Hydrophobicity',
           'Hydrophilicity',
           'Neutral',
           'Hydroxylic',
           'Sulphur content',
           'Secondary Structure(Helix)',
           'Secondary Structure(Strands)',
           'Secondary Structure(Coil)',
           'Solvent Accessibilty (Buried)',
           'Solvent Accesibilty(Exposed)',
           'Solvent Accesibilty(Intermediate)',
           'Tiny',
           'Small',
           'Large',
           'z1',
           'z2',
           'z3',
           'z4',
           'z5'];


def encode(peptide):
    #letter = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'};
    l=len(peptide)
    encoded=np.zeros(l)
    for i in range(l):
        if(peptide[i]=='A'):
            encoded[i] = 0
        elif(peptide[i]=='C'):
            encoded[i] = 1
        elif(peptide[i]=='D'):
            encoded[i] = 2
        elif(peptide[i]=='E'):
            encoded[i] = 3
        elif(peptide[i]=='F'):
            encoded[i] = 4
        elif(peptide[i]=='G'):
            encoded[i] = 5
        elif(peptide[i]=='H'):
            encoded[i] = 6
        elif(peptide[i]=='I'):
            encoded[i] = 7
        elif(peptide[i]=='K'):
            encoded[i] = 8
        elif(peptide[i]=='L'):
            encoded[i] = 9
        elif(peptide[i]=='M'):
            encoded[i] = 10
        elif(peptide[i]=='N'):
            encoded[i] = 11
        elif(peptide[i]=='P'):
            encoded[i] = 12
        elif(peptide[i]=='Q'):
            encoded[i] = 13
        elif(peptide[i]=='R'):
            encoded[i] = 14
        elif(peptide[i]=='S'):
            encoded[i] = 15
        elif(peptide[i]=='T'):
            encoded[i] = 16
        elif(peptide[i]=='V'):
            encoded[i] = 17
        elif(peptide[i]=='W'):
            encoded[i] = 18
        elif(peptide[i]=='Y'):
            encoded[i] = 19
        else:
            print('Wrong residue!')
    return encoded


def lookup(peptide,featureNum):
    l=len(peptide)
    peptide = list(peptide)
    out=np.zeros(l)
    peptide_num = encode(peptide)

    for i in range(l):
        out[i] = PCP[peptide_num[i]][featureNum]
    return sum(out)


def pcp(seq):

    l = len(seq)

    rows = PCP.shape[0] # Number of features in our reference table
    col = 20  # Denotes the 20 amino acids

    seq=[seq[i].upper() for i in range(l)]
    sequenceFeature = {}


    for i in range(l): # Loop to iterate over each sequence
        nfeatures = rows
        for j in range(nfeatures): #Loop to iterate over each feature
            featureVal = lookup(seq[i],j)
            if(len(seq[i])!=0):
                sequenceFeature[headers[j]]=(round(featureVal/len(seq[i]),3))
            else:
                sequenceFeature[headers[j]]='NaN'
    print(sequenceFeature)
    return sequenceFeature

#
# # In[68]:
#
#
# '''
#
# unction Name: phyChem
# Description: Gives 30 physico-chemical properties of a sequence
#
# Function prototype: phyChem(file,mode,m,n)
#
# Input:
# @file: an input csv file with multiple sequences
# @mode(optional, default = 'all'):
#     Values possible:
#         1) (default)'all' : to get features of entire protein
#         2) 'NT' : to get the features of first n N-Terminal residues
#         3) 'CT' : to get the features of last n C-Terminal residues
#         4) 'rest' : to get the features of a sub-sequence from mth position to nth position(both inclusive)
# @m(optional(mandatory if 'rest' is chosen, default=0): m is the start position of residue
# @n(optional, default = '0'): n is the number of residues you want from desired terminal or end point (if 'rest' is chosen)
#
#
# Output:
# A matrix (csv file) of dimension (m x 30) containing sequences as rows and their 30 physico-chemical properties as columns
# where m = number of sequences (each sequence separated by comma)
#
# '''
#
#
# '''
#
# Function Name: phyChem
# Description: Gives 30 physico-chemical properties of a sequence
#
# Function prototype: phyChem(file,mode,m,n)
#
# Input:
# @file: an input csv file with multiple sequences
# @mode(optional, default = 'all'):
#     Values possible:
#         1) (default)'all' : to get features of entire protein
#         2) 'NT' : to get the features of first n N-Terminal residues
#         3) 'CT' : to get the features of last n C-Terminal residues
#         4) 'rest' : to get the features of a sub-sequence from mth position to nth position(both inclusive)
# @m(optional(mandatory if 'rest' is chosen, default=0): m is the start position of residue
# @n(optional, default = '0'): n is the number of residues you want from desired terminal or end point (if 'rest' is chosen)
#
#
# Output:
# A matrix (csv file) of dimension (m x 30) containing sequences as rows and their 30 physico-chemical properties as columns
# where m = number of sequences (each sequence separated by comma)
#
# '''
#
#
# def pcp_wp(file,outt,mode='all',m=0,n=0):
#
#     if(type(file) == str):
#         seq = pd.read_csv(file,header=None, sep=',');
#         seq=seq.T
#         seq[0].values.tolist()
#         seq=seq[0];
#     #elif(type(file)==str && len(file))
#     else:
#         seq  = file;
#
#     l = len(seq);
#
#     newseq = [""]*l; # To store the trimmed sequence
#     #print('Original Sequence');
#     #print(seq)
#
#
#     for i in range(0,l):
#
#         #if(n<len(seq[i])):
#         l = len(seq[i]);
#
#         if(mode=='NT'):
#             n=m;
#             if(n!=0):
#                 newseq[i] = seq[i][0:n];
#
#             elif(n>l):
#                 print('Warning! Sequence',i,"'s size is less than n. The output table would have NaN for this sequence");
#
#             else:
#                 print('Value of n is mandatory, it cannot be 0')
#                 break;
#
#         elif(mode=='CT'):
#             n=m;
#             if(n!=0):
#                 newseq[i] = seq[i][(len(seq[i])-n):]
#
#             elif(n>l):
#                 print('WARNING: Sequence',i+1,"'s size is less than the value of n given. The output table would have NaN for this sequence");
#
#
#             else:
#                 print('Value of n is mandatory, it cannot be 0')
#                 break
#
#         elif(mode=='all'):
#             newseq = seq
#
#         elif(mode=='rest'):
#             if(m==0):
#                 print('Kindly provide start index for rest, it cannot be 0');
#                 break
#
#             else:
#                 if(n<=len(seq[i])):
#                     newseq[i] = seq[i][m:(len(seq[i])-n)]
#                 '''elif(n>len(seq[i])):
#                     newseq[i] = seq[i][m-1:len(seq[i])]
#                     print('WARNING: Since input value of n for sequence',i+1,'is greater than length of the protein, entire sequence starting from m has been considered')'''
# #
# #         else:
# #             print("Wrong Mode. Enter 'NT', 'CT','all' or 'rest'");
#
#
#     output = pcp(newseq,outt);
#     return output



if __name__ == '__main__':
    pcp('KRVLLLDNLSDYIKPGMSVEAIQGIIASMKSDYEDRVDDYIIKNAELSKERRDISKKLKVMGE')
    pcp('VVLSDQGALFEPCSTQDIACLSRATQQFLEKACRGVPEYDIRPIDPLIISSLDVAAYDDIGLIFHFKNLNITGLKNQKISDFRMDTTRKSVLLKTQADLNVVADVVIELSKQSKSFAGVMNIQASIIGGAKYSYDLQDDSKGVKHFEVGQETISCESIGEPAVNLNPELADALLKDPDTTHYRKDYEAHRVSIRQ')