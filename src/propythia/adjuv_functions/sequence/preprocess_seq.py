"""
#####################################################################################
Preprocessing the amino acid sequences.

Authors: Miguel Barros

Date: 02/2022

Email:

#####################################################################################

"""
AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]


def protein_preprocessing(ProteinSequence, B :str ='N', Z : str = 'Q', U :str = 'C', O: str = 'K', J : str = 'I', X :str = ''):
	'''
	Edits a protein sequence by replacing aminoacids like Asparagine (B),  Glutamine(Z), Selenocysteine (U) and
         Pyrrolysine (O), by default they are replaced for the closest aminoacid residue if is present in the sequence, Asparagine (N), Glutamine (Q),
         Cysteinen(C) and Lysine (K), respectively. It also removes the ambiguous aminoacid (X) and alters the ambiguous
         aminoacid (J), by default it is replaced for Isoleucine (I). Use with caution.

	:param ProteinSequence: Protein sequence
	:param B: One-letter aminoacid code to replace Asparagine (B). Default amino acid is Asparagine (N)
	:param Z: One-letter aminoacid code to replace Glutamine(Z). Default amino acid is Glutamine (Q)
	:param U: One-letter aminoacid code to replace Selenocysteine (U). Default amino acid is Cysteinen(C)
	:param O: One-letter aminoacid code to replace Pyrrolysine (O). Default amino acid is Lysine (K)
	:param J: One-letter aminoacid code to replace ambiguous aminoacid (J). Default amino acid is Isoleucine (I)
	:param X: One-letter aminoacid code to replace ambiguous aminoacid (X), by default it is removed.
	:return: transformed protein sequence
	'''
	Seq = ProteinSequence.replace('B', B).replace('Z', Z).replace('U',U).replace('O', O).replace('X', X)\
		.replace('J', J)
	return Seq

