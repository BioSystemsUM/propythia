# -*- coding: utf-8 -*-
"""
#####################################################################################
Checking whether the input protein sequence is valid amino acid or not

Authors: Ana Marta Sequeira

Date:

Email:

#####################################################################################

"""
AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

def protein_check(ProteinSequence):
	"""
	Check whether the protein sequence is a valid amino acid sequence or not

	:param ProteinSequence: protein sequence
	:return: length of protein if no problem, 0 if protein is not valid
	"""
	global flag
	num_pro=len(ProteinSequence)
	for i in ProteinSequence:
		if i not in AALetter:
			flag=0
			break
		else:
			flag=num_pro
	return flag

def protein_preprocessing_X(ProteinSequence):
	'''
	Edits a protein sequence by replacing aminoacids like Asparagine (B),  Glutamine(G), Selenocysteine (U) and
	 Pyrrolysine (O) by an ambiguous aminoacid (X). It also alters the ambiguos J by X.

	:param ProteinSequence: Protein sequence
	:return: transformed protein sequence
	'''
	Seq = ProteinSequence.replace('B', 'X').replace('Z', 'X').replace('U','X').replace('O', 'X').replace('J', 'X')
	return Seq

def protein_preprocessing_20AA(ProteinSequence):
	'''
	Edits a protein sequence by replacing aminoacids like Asparagine (B),  Glutamine(G), Selenocysteine (U) and
	 Pyrrolysine (O) for the closest aminoacid residue if is present in the sequence, Asparagine (N), Glutamine (Q),
	 Cysteinen(C) and Lysine (K), respectively. It also removes the ambiguous aminoacid (X) and alters the ambiguous
	 aminoacid (J) to a Isoleucine (I). Use with caution.

	:param ProteinSequence: Protein sequence
	:return: transformed protein sequence
	'''
	Seq = ProteinSequence.replace('B', 'N').replace('Z', 'Q').replace('U','C').replace('O', 'K').replace('X', '')\
		.replace('J', 'I')
	return Seq

def protein_preprocessing_removeX(ProteinSequence):
	'''
	Edits a protein sequence by removing the ambiguous aminoacid (X).

	:param ProteinSequence: Protein sequence
	:return: transformed protein sequence
	'''
	Seq = ProteinSequence.replace('X', '')
	return Seq

if __name__ == "__main__":

	protein_inv="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDASU"
	print(protein_check(protein_inv))
	protein='MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNL'
	print(protein_check(protein))