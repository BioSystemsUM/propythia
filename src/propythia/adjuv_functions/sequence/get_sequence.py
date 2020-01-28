# -*- coding: utf-8 -*-
"""
################################################################################################

This file contains functions to get protein sequences.

To download a protein sequence from the uniprot (http://www.uniprot.org/) website,
just provide a protein ID or a txt file with uniprot ID, obtaining a txt file with the protein sequences.

Authors:

Date:

Email:

################################################################################################
"""

import urllib.request, urllib.parse, urllib.error
import string


def get_protein_sequence(protein_id):
    """
    Get the protein sequence from the uniprot website by ID.
    :param protein_id: ProteinID is a string indicating ID such as "P48039".
    :return: str protein sequence
    """

    ID=str(protein_id)
    url='http://www.uniprot.org/uniprot/'+ID+'.fasta'
    #print(url)
    localfile=urllib.request.urlopen(url)

    temp=localfile.readlines()
    res=''
    for i in range(1,len(temp)):
        res=res+str.strip(str(temp[i]))
    AALetter=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    proteinsequence=''.join(c for c in res if c in AALetter)

    return proteinsequence


def get_protein_sequence_from_txt(path, openfile, savefile):
    """
    Get the protein sequence from the uniprot website by the file containing ID.
    :param path: path is a directory path containing the ID file (e.g. "/home/protein/")
    :param openfile: ID file  (e.g. "proteinID.txt")
    :param savefile: file saving the obtained protein sequences (e.g. "protein.txt")
    :return: file txt with protein sequences
    """

    f1=open(path+savefile,'wb')
    f2=open(path+openfile,'r')
    #	res=[]
    for index,i in enumerate(f2):

        itrim=str.strip(i)
        if itrim == "":
            continue
        else:
            temp=get_protein_sequence(itrim)
            print("--------------------------------------------------------")
            print("The %d protein sequence has been downloaded!" %(index+1))
            print(temp)
            f1.write(str.encode(temp+'\n'))
            print("--------------------------------------------------------")
    #		res.append(temp+'\n')
    #	f1.writelines(res)
    f2.close()
    f1.close()
    return 'end'


if __name__ == '__main__':

    uniprot=get_protein_sequence('P08172')
    print(uniprot)

    result_ID=get_protein_sequence_from_txt(r"", "data/target.txt", "data/result.txt")
    print(result_ID)






