# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
##############################################################################
This function gives binary profile of residues for all the  sequences for 25 phychem feature

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

The code is based on the package Pfeature:
Pande, Akshara & Patiyal, Sumeet & Lathwal, Anjali & Arora, Chakit & Kaur, Dilraj & Dhall, Anjali & Mishra, Gaurav & Kaur,
Harpreet & Sharma, Neelam & Jain, Shipra & Usmani, Salman & Agrawal, Piyush & Kumar, Rajesh & Kumar, Vinod & Raghava, Gajendra.
(2019). Computing wide range of protein/peptide features from their sequence and structure. 10.1101/599126.


It returns a dictionary form with the values

Authors: Ana Marta Sequeira

Date: 05/2019

Email:

##############################################################################
"""

import pandas as pd
import numpy as np
import os
import sys


def init(path=None, index=''):
    """
    Read in files. You need to run this (once) before you can
    access any records. If the files are not within the current directory,
    you need to specify the correct directory path.
    :param path:
    :param index:
    :return:
    """
    index = str(index)
    if path is None:
        for path in [os.path.split(__file__)[0]]:
            if os.path.exists(os.path.join(path, index)):
                break
        # print('path =', path, file=sys.stderr)

    doc=''
    if index == 'bonds.csv':
        doc=(path + '/data/bonds.scv')
    if index == 'PhysicoChemical.csv':
        doc=(path + '/data/PhysicoChemical.csv')
    return doc


doc=init(index='PhysicoChemical.csv')
# Finding physico-chemical property of a vector of polypeptides
PCP = pd.read_csv(doc, header=None)

#headerCT = ['Positively charged_CT_1', 'Positively charged_CT_2', 'Positively charged_CT_3', 'Positively charged_CT_4', 'Positively charged_CT_5', 'Positively charged_CT_6', 'Positively charged_CT_7', 'Positively charged_CT_8', 'Positively charged_CT_9', 'Positively charged_CT_10', 'Negatively charged_CT_1', 'Negatively charged_CT_2', 'Negatively charged_CT_3', 'Negatively charged_CT_4', 'Negatively charged_CT_5', 'Negatively charged_CT_6', 'Negatively charged_CT_7', 'Negatively charged_CT_8', 'Negatively charged_CT_9', 'Negatively charged_CT_10', 'Neutral charged_CT_1', 'Neutral charged_CT_2', 'Neutral charged_CT_3', 'Neutral charged_CT_4', 'Neutral charged_CT_5', 'Neutral charged_CT_6', 'Neutral charged_CT_7', 'Neutral charged_CT_8', 'Neutral charged_CT_9', 'Neutral charged_CT_10', 'Polarity_CT_1', 'Polarity_CT_2', 'Polarity_CT_3', 'Polarity_CT_4', 'Polarity_CT_5', 'Polarity_CT_6', 'Polarity_CT_7', 'Polarity_CT_8', 'Polarity_CT_9', 'Polarity_CT_10', 'Non polarity_CT_1', 'Non polarity_CT_2', 'Non polarity_CT_3', 'Non polarity_CT_4', 'Non polarity_CT_5', 'Non polarity_CT_6', 'Non polarity_CT_7', 'Non polarity_CT_8', 'Non polarity_CT_9', 'Non polarity_CT_10', 'Aliphaticity_CT_1', 'Aliphaticity_CT_2', 'Aliphaticity_CT_3', 'Aliphaticity_CT_4', 'Aliphaticity_CT_5', 'Aliphaticity_CT_6', 'Aliphaticity_CT_7', 'Aliphaticity_CT_8', 'Aliphaticity_CT_9', 'Aliphaticity_CT_10', 'Cyclic_CT_1', 'Cyclic_CT_2', 'Cyclic_CT_3', 'Cyclic_CT_4', 'Cyclic_CT_5', 'Cyclic_CT_6', 'Cyclic_CT_7', 'Cyclic_CT_8', 'Cyclic_CT_9', 'Cyclic_CT_10', 'Aromaticity_CT_1', 'Aromaticity_CT_2', 'Aromaticity_CT_3', 'Aromaticity_CT_4', 'Aromaticity_CT_5', 'Aromaticity_CT_6', 'Aromaticity_CT_7', 'Aromaticity_CT_8', 'Aromaticity_CT_9', 'Aromaticity_CT_10', 'Acidicity_CT_1', 'Acidicity_CT_2', 'Acidicity_CT_3', 'Acidicity_CT_4', 'Acidicity_CT_5', 'Acidicity_CT_6', 'Acidicity_CT_7', 'Acidicity_CT_8', 'Acidicity_CT_9', 'Acidicity_CT_10', 'Basicity_CT_1', 'Basicity_CT_2', 'Basicity_CT_3', 'Basicity_CT_4', 'Basicity_CT_5', 'Basicity_CT_6', 'Basicity_CT_7', 'Basicity_CT_8', 'Basicity_CT_9', 'Basicity_CT_10', 'Neutral (ph)_CT_1', 'Neutral (ph)_CT_2', 'Neutral (ph)_CT_3', 'Neutral (ph)_CT_4', 'Neutral (ph)_CT_5', 'Neutral (ph)_CT_6', 'Neutral (ph)_CT_7', 'Neutral (ph)_CT_8', 'Neutral (ph)_CT_9', 'Neutral (ph)_CT_10', 'Hydrophobicity_CT_1', 'Hydrophobicity_CT_2', 'Hydrophobicity_CT_3', 'Hydrophobicity_CT_4', 'Hydrophobicity_CT_5', 'Hydrophobicity_CT_6', 'Hydrophobicity_CT_7', 'Hydrophobicity_CT_8', 'Hydrophobicity_CT_9', 'Hydrophobicity_CT_10', 'Hydrophilicity_CT_1', 'Hydrophilicity_CT_2', 'Hydrophilicity_CT_3', 'Hydrophilicity_CT_4', 'Hydrophilicity_CT_5', 'Hydrophilicity_CT_6', 'Hydrophilicity_CT_7', 'Hydrophilicity_CT_8', 'Hydrophilicity_CT_9', 'Hydrophilicity_CT_10', 'Neutral_CT_1', 'Neutral_CT_2', 'Neutral_CT_3', 'Neutral_CT_4', 'Neutral_CT_5', 'Neutral_CT_6', 'Neutral_CT_7', 'Neutral_CT_8', 'Neutral_CT_9', 'Neutral_CT_10', 'Hydroxylic_CT_1', 'Hydroxylic_CT_2', 'Hydroxylic_CT_3', 'Hydroxylic_CT_4', 'Hydroxylic_CT_5', 'Hydroxylic_CT_6', 'Hydroxylic_CT_7', 'Hydroxylic_CT_8', 'Hydroxylic_CT_9', 'Hydroxylic_CT_10', 'Sulphur content_CT_1', 'Sulphur content_CT_2', 'Sulphur content_CT_3', 'Sulphur content_CT_4', 'Sulphur content_CT_5', 'Sulphur content_CT_6', 'Sulphur content_CT_7', 'Sulphur content_CT_8', 'Sulphur content_CT_9', 'Sulphur content_CT_10', 'Secondary Structure(Helix)_CT_1', 'Secondary Structure(Helix)_CT_2', 'Secondary Structure(Helix)_CT_3', 'Secondary Structure(Helix)_CT_4', 'Secondary Structure(Helix)_CT_5', 'Secondary Structure(Helix)_CT_6', 'Secondary Structure(Helix)_CT_7', 'Secondary Structure(Helix)_CT_8', 'Secondary Structure(Helix)_CT_9', 'Secondary Structure(Helix)_CT_10', 'Secondary Structure(Strands)_CT_1', 'Secondary Structure(Strands)_CT_2', 'Secondary Structure(Strands)_CT_3', 'Secondary Structure(Strands)_CT_4', 'Secondary Structure(Strands)_CT_5', 'Secondary Structure(Strands)_CT_6', 'Secondary Structure(Strands)_CT_7', 'Secondary Structure(Strands)_CT_8', 'Secondary Structure(Strands)_CT_9', 'Secondary Structure(Strands)_CT_10', 'Secondary Structure(Coil)_CT_1', 'Secondary Structure(Coil)_CT_2', 'Secondary Structure(Coil)_CT_3', 'Secondary Structure(Coil)_CT_4', 'Secondary Structure(Coil)_CT_5', 'Secondary Structure(Coil)_CT_6', 'Secondary Structure(Coil)_CT_7', 'Secondary Structure(Coil)_CT_8', 'Secondary Structure(Coil)_CT_9', 'Secondary Structure(Coil)_CT_10', 'Solvent Accessibilty (Buried)_CT_1', 'Solvent Accessibilty (Buried)_CT_2', 'Solvent Accessibilty (Buried)_CT_3', 'Solvent Accessibilty (Buried)_CT_4', 'Solvent Accessibilty (Buried)_CT_5', 'Solvent Accessibilty (Buried)_CT_6', 'Solvent Accessibilty (Buried)_CT_7', 'Solvent Accessibilty (Buried)_CT_8', 'Solvent Accessibilty (Buried)_CT_9', 'Solvent Accessibilty (Buried)_CT_10', 'Solvent Accesibilty(Exposed)_CT_1', 'Solvent Accesibilty(Exposed)_CT_2', 'Solvent Accesibilty(Exposed)_CT_3', 'Solvent Accesibilty(Exposed)_CT_4', 'Solvent Accesibilty(Exposed)_CT_5', 'Solvent Accesibilty(Exposed)_CT_6', 'Solvent Accesibilty(Exposed)_CT_7', 'Solvent Accesibilty(Exposed)_CT_8', 'Solvent Accesibilty(Exposed)_CT_9', 'Solvent Accesibilty(Exposed)_CT_10', 'Solvent Accesibilty(Intermediate)_CT_1', 'Solvent Accesibilty(Intermediate)_CT_2', 'Solvent Accesibilty(Intermediate)_CT_3', 'Solvent Accesibilty(Intermediate)_CT_4', 'Solvent Accesibilty(Intermediate)_CT_5', 'Solvent Accesibilty(Intermediate)_CT_6', 'Solvent Accesibilty(Intermediate)_CT_7', 'Solvent Accesibilty(Intermediate)_CT_8', 'Solvent Accesibilty(Intermediate)_CT_9', 'Solvent Accesibilty(Intermediate)_CT_10', 'Tiny_CT_1', 'Tiny_CT_2', 'Tiny_CT_3', 'Tiny_CT_4', 'Tiny_CT_5', 'Tiny_CT_6', 'Tiny_CT_7', 'Tiny_CT_8', 'Tiny_CT_9', 'Tiny_CT_10', 'Small_CT_1', 'Small_CT_2', 'Small_CT_3', 'Small_CT_4', 'Small_CT_5', 'Small_CT_6', 'Small_CT_7', 'Small_CT_8', 'Small_CT_9', 'Small_CT_10', 'Large_CT_1', 'Large_CT_2', 'Large_CT_3', 'Large_CT_4','Large_CT_5', 'Large_CT_6', 'Large_CT_7', 'Large_CT_8', 'Large_CT_9', 'Large_CT_10'];
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
           'Large']


def encode(peptide):
    """
    encode AA in sequence
    :param peptide: protein sequence
    :return: protein sequence with encoded AA
    """
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


def lookup(peptide, feature_num):
    """
    :param peptide: protein sequence
    :param feature_num: number of the feature to be calculated
    :return: value of property for each AA in sequence
    """
    l=len(peptide)
    peptide = list(peptide)
    out=[]
    peptide_num = encode(peptide)

    for i in range(l):
        #out[i] = PCP[peptide_num[i]][featureNum]
        out.append(PCP[peptide_num[i]][feature_num])
    return out


def binary_profile_all(seq, feature_numb):
    """
    binary profile of physicalchemical characterisitics choosed of a sequence
    :param seq: protein sequence
    :param feature_numb: number of the feature to be calculated
    :return: value of property for each AA in sequence
    """
    l = len(seq)
    #print('length is',l)
    seq=[seq[i].upper() for i in range(l)]
    bin_prof = []
    for i in range(0,l):
        temp = lookup(seq[i], feature_numb)
        bin_prof.append(temp)
    out = pd.DataFrame(bin_prof)

    return bin_prof


def bin_pc_wp(sequence, feature_numb):
    """
    Binary profile of a sequence based on physicalchemical characterisitics choosed
    :param sequence: protein sequence
    :param feature_numb: list with numbers of the features to be calculated
    :return: dictionary with binary profile of a sequence for a list of desired features
    """
    res={}
    for feature in feature_numb:
        binprof = binary_profile_all(sequence,feature)
        for j in range(len(binprof)):
            name_feature='bin_proper_'+str(feature)+'_'+str(j)
            res[name_feature]=binprof[j][0]

    return res


if __name__=="__main__":
    print(bin_pc_wp('AYARY',[24]))
    print(bin_pc_wp('AAACPPPQSTY',[24,5,2,0]))
    print(bin_pc_wp('APACMPPQSTY',[24,5,2,0]))


