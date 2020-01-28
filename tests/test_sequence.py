"""
##############################################################################

File containing tests functions to check if all functions from sequence module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
from propythia.sequence import ReadSequence

def test_sequence():
    sequence=ReadSequence() #create the object to read sequence

    print('read sequence from uniprot id')
    ps=sequence.get_protein_sequence_from_id('P48039') #from uniprot id

    print('read sequence from sequence')
    ps_string=sequence.read_protein_sequence("MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN")#from string


    print('check protein')
    protein_inv="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDASU"
    sequence.checkprotein(protein_inv)
    protein='MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNL'
    sequence.checkprotein(protein)


    print('obtain sequences with equal size')
    list_reverse_n=(sequence.get_sized_seq(['AAAANNDAKMAPSSAA', 'AAAANNDAKMAPSSAAAAAAAAAAAAA', 'AAAAKMAAA', 'AAAANNDAKMAPSSAAAAAAAAAAAAAAAAANNDAKMAPSSAAAAAAAAAAAAA', 'Z'], 10, 0))
    print(list_reverse_n)
    list_in_c=sequence.get_sized_seq(['AAAANNDAKMAPSSAA', 'AAAANNDAKMAPSSAAAAAAAAAAAAA', 'AAAAKMAAA', 'AAAANNDAKMAPSSAAAAAAAAAAAAAAAAANNDAKMAPSSAAAAAAAAAAAAA', 'Z'], 0, 10)
    print(list_in_c)
    list_reverse_cn=(sequence.get_sized_seq(['AAAANNDAKMAPSSAA', 'AAAANNDAKMAPSSAAAAAAAAAAAAA', 'AAAAKMAAA', 'AAAANNDAKMAPSSAAAAAAAAAAAAAAAAANNDAKMAPSSAAAAAAAAAAAAA', 'Z'], 5, 5, 2))
    print(list_reverse_cn)
    print((sequence.get_sized_seq('AAVFNDRAT', 5, 5, 2)))
    print((sequence.get_sized_seq('AAVFNDRAT', 15, 5, 2)))

    print('Generating subsequences')
    protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"

    list_subseq,indices=sequence.get_sub_seq_sliding_window (protein, window_size=20, gap=10, index=True)
    print(list_subseq,indices)
    subseq_to_aa=sequence.get_sub_seq_to_aa(protein, ToAA='S', window=5)
    print(subseq_to_aa)
    subseq_split=sequence.get_sub_seq_split(protein, number_of_subseq=5)
    print(subseq_split)
    subseq_terminals=sequence.get_sub_seq_terminals(protein, N_terminal=5, C_terminal=5, rest=True)
    print(subseq_terminals)


if __name__ == "__main__":
    test_sequence()