"""
#####################################################################################
Preprocessing the amino acid sequences.

Authors: Miguel Barros

Date: 02/2022

Email:

#####################################################################################

"""
AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

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
