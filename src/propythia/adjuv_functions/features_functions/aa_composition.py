# -*- coding: utf-8 -*-
"""

###############################################################################

The module is used for computing the composition of amino acids, dipetide and 

3-mers (tri-peptide) for a given protein sequence. You can get 8420 descriptors

for a given protein sequence. You can freely use and distribute it. If you hava 

any problem, you could contact with us timely!

References:

[1]: Reczko, M. and Bohr, H. (1994) The DEF data base of sequence based protein

fold class predictions. Nucleic Acids Res, 22, 3616-3619.

[2]: Hua, S. and Sun, Z. (2001) Support vector machine approach for protein

subcellular localization prediction. Bioinformatics, 17, 721-728.


[3]:Grassmann, J., Reczko, M., Suhai, S. and Edler, L. (1999) Protein fold class

prediction: new methods of statistical classification. Proc Int Conf Intell Syst Mol

Biol, 106-112.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.3.27

Email: oriental-cds@163.com

Altered and converted to python 3.6 for Ana Marta Sequeira 05/2019
###############################################################################
"""

import re

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
#############################################################################################


def calculate_aa_composition(protein_sequence):
	"""
	Calculate the composition of Amino acids for a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing the composition of 20 amino acids.
	"""

	length_sequence=len(protein_sequence)
	result={}
	for i in AALetter:
		result[i] = round(float(protein_sequence.count(i)) / length_sequence * 100, 3)
	return result
#############################################################################################


def calculate_dipeptide_composition(protein_sequence):
	"""
	Calculate the composition of dipeptidefor a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing the composition of 400 dipeptides.
	"""
	length_sequence=len(protein_sequence)
	result={}
	for i in AALetter:
		for j in AALetter:
			dipeptide=i+j
			result[dipeptide]=round(float(protein_sequence.count(dipeptide)) / (length_sequence - 1) * 100, 2)
	return result
#############################################################################################


def getkmers():
	"""
	Get the amino acid list of 3-mers.
	:return: result is a list form containing 8000 tri-peptides.
	"""
	kmers = list()
	for i in AALetter:
		for j in AALetter:
			for k in AALetter:
				kmers.append(i+j+k)
	return kmers

#############################################################################################


def get_spectrum_dict(protein_sequence):
	"""
	Calculate the spectrum of 3-mers for a given protein.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing the composition values of 8000
	"""

	result = {}
	kmers = getkmers()
	for i in kmers:
		result[i]=len(re.findall(i, protein_sequence))
	return result

#############################################################################################


def calculate_aa_tripeptide_composition(protein_sequence):
	"""
	Calculate the composition of AADs, dipeptide and 3-mers for a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing all composition values of AADs, dipeptide and 3-mers (8420).
	"""
	result={}
	result.update(calculate_aa_composition(protein_sequence))
	result.update(calculate_dipeptide_composition(protein_sequence))
	result.update(get_spectrum_dict(protein_sequence))

	return result


if __name__ == "__main__":

	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"

	AAC=calculate_aa_composition(protein)
	print(AAC)
	DIP=calculate_dipeptide_composition(protein)
	print(DIP)
	spectrum=get_spectrum_dict(protein)
	print(spectrum)
	res=calculate_aa_tripeptide_composition(protein)
	print(len(res))


