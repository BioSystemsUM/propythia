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

if __name__ == "__main__":

	protein_inv="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDASU"
	print(protein_check(protein_inv))
	protein='MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNL'
	print(protein_check(protein))