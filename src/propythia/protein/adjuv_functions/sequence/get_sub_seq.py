# -*- coding: utf-8 -*-
"""
#####################################################################################

Allows to generate subsequences from only one sequence. The output is a list of sequences.
(one sequence ----> list with subsequences)

The subsequences can be:
	1) sliding window of the protein given. It will generate a list of n sequences with
	lenght equal to the value of window and spaced a gap value. It can or not retrieve
	the indices of location of the subsequence in the original sequence.
	This can be useful for example to screen specific sites in the protein with machine learning.

	2)Split the total protein into a set of segments around specific aminoacid.Given a
	specific window size p, we can obtain all segments of length equal to (2*p+1).
	It can be useful for example in the prediction of functional sites (e.g.,methylation) of protein.

	3)Split the original sequence in a user specified number of subsequences. Divide the sequence in n
	equal lenght (when possible) chunks

	4)Divide the sequence in the N terminal and  C terminal with sizes defined by the user. It returns
	a list with N and C terminal and the rest of the sequence(if user choose).
	By default the N terminal is considerer to be the beggining of the sequence (left)


Authors:Ana Marta Sequeira

Date:

Email:

#####################################################################################

"""

import re
import string

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
#############################################################################################

def sub_seq_sliding_window(ProteinSequence,window_size=20,gap=1,index=True):
	"""
	sliding window of the protein given. It will generate a list of n sequences with
	lenght equal to the value of window and spaced a gap value. It can or not retrieve
	the indices of location of the subsequence in the original sequence.

	:param ProteinSequence: protein sequence
	:param window_size: number of aminoacids to considerer, lenght of the subsequence. for default 20
	:param gap: gap size of the search of windows in sequence. default 1
	:param index: if true, return the indices of location of the subsequence in the original sequence
	:return: list with subsequences generated with or without a list of tuples with location of subsequences
	 in original sequence
	"""

	m=len(ProteinSequence)
	n=int(window_size)
	list_of_sequences=[]
	indices=[]

	for i in range(0,m-(n-1),gap):
		list_of_sequences.append(ProteinSequence[i:i+n])
		indices.append((i,i+n))
		i+=1
	if index: return list_of_sequences,indices
	else: return list_of_sequences


def sub_seq_to_aa(ProteinSequence, ToAA, window):
	"""
	Get all 2*window+1 sub-sequences whose center is ToAA in a protein

	:param ProteinSequence:  protein sequence
	:param ToAA: central (query point) amino acid in the sub-sequence
	:param window:  span (number of amnoacids to go front and back from the ToAA
	:return: list form containing sub-sequences
	"""

	if ToAA not in AALetter:
		ToAA=ProteinSequence[1]
	
	Num=len(ProteinSequence)
	seqiter=re.finditer(ToAA,ProteinSequence)
	AAindex=[]
	for i in seqiter:
		AAindex.append(i.end())
	
	result=[]
	for i in AAindex:
		if i-window>0 and Num-i+1-window>0:
			temp=ProteinSequence[i-window-1:i+window]
			result.append(temp)
	
	return result


def sub_seq_split(seq,number_of_subseq):
	"""
	Split the originl seq in n number of subsequences.

	:param seq: protein sequence
	:param number_of_subseq: number of subsequences to divide the original seq
	:return: list with number_of_sequences sequences.

	"""

	avg = len(seq) / float(number_of_subseq)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg

	return out


def sub_seq_terminals(seq, N_terminal=5, C_terminal=5,rest=True):
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

	result=[]

	nterm=seq[:N_terminal]
	result.append(nterm)

	if C_terminal!=0:
		cterm=seq[-C_terminal:]
		result.append(cterm)

	if rest:
		rest_list=seq[N_terminal:-C_terminal]
		result.append(rest_list)

	return result


if __name__=="__main__":
	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"

	list_subseq,indices=sub_seq_sliding_window(protein,window_size=20,gap=10,index=True)
	print(list_subseq,indices)


	subseq_to_aa=sub_seq_to_aa(protein,ToAA='D',window=4)
	print(subseq_to_aa)


	subseq_split=sub_seq_split(protein,number_of_subseq=5)
	print(subseq_split)


	subseq_terminals=sub_seq_terminals(protein, N_terminal=5, C_terminal=5,rest=True)
	print(subseq_terminals)




