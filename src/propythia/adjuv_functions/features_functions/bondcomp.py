# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
This function gives the sum of the bond composition for each type of bond
For bond composition four types of bonds are
considered total number of bonds (including aromatic), hydrogen bond, single bond and double
bond. The number of values for each kind of bond is provided as bonds.csv file

The code is based on the package Pfeature :
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
        doc=(path + '/data/bonds.csv')
    if index == 'PhysicoChemical.csv':
        doc=(path + '/data/PhysicoChemical.csv')
    return doc


doc=init(index='bonds.csv')
#Finding physico-chemical property of a vector of polypeptides
bonds = pd.read_csv(doc, header=None)


def boc_wp(seq):
    """
    Sum of the bond composition for each type of bond: total number of bonds (including aromatic), hydrogen bond,
    single bond and double
    :param seq: protein sequence
    :return: dictionary with number of total, hydrogen, single and double bonds
    """
    tota = []
    hy = []
    Si = []
    Du = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    bb = {}

    df = seq

    for i in range(0,len(df)) :
        tot = 0
        h = 0
        S = 0
        D = 0
        tota.append([i])
        hy.append([i])
        Si.append([i])
        Du.append([i])
        for j in range(0,len(df[i])) :
            temp = df[i][j]
            for k in range(0,len(bonds)) :
                if bonds.iloc[:,0][k] == temp :
                    tot = tot + int(bonds.iloc[:,1][k])
                    h = h + int(bonds.iloc[:,2][k])
                    S = S + int(bonds.iloc[:,3][k])
                    D = D + int(bonds.iloc[:,4][k])
        tota[i].append(tot)
        hy[i].append(h)
        Si[i].append(S)
        Du[i].append(D)
    for m in range(0,len(df)) :
        b1.append(tota[m][1])
        b2.append(hy[m][1])
        b3.append(Si[m][1])
        b4.append(Du[m][1])

    #IF BOND COMPOSITION SEPARATEDLY BY AA RESIDUE JUST TAKE OFF SUM (AND WILL GIVE 4*LEN(SEQ)
    bb["tot"] = sum(b1)
    bb["hydrogen"] = sum(b2)
    bb["single"] = sum(b3)
    bb["double"] = sum(b4)
    
    return bb


if __name__ == '__main__':
    print(boc_wp('MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN'))
    print(boc_wp('MALPNAVIAAAALSVYRNKKLRN'))
    print(boc_wp('MQGNGSPALLNSRRRRRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN'))

