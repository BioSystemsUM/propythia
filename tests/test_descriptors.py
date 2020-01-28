"""
##############################################################################

File containing tests functions to check if all functions from descriptors module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""




from propythia.descriptors import Descriptor
from propythia.sequence import ReadSequence
import timeit


def test_descriptors():
    sequence=ReadSequence() #create the object to read sequence

    ps=sequence.get_protein_sequence_from_id('P48039') #from uniprot id
    protein = Descriptor(ps) # creating object to calculate descriptors

    ps_string=sequence.read_protein_sequence("MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN")#from string
    protein2=Descriptor(ps_string)


    print('tests protein1')
    start = timeit.default_timer()
    test1=protein.get_all() #all except tripeptide and binaries representations
    test1_2=protein.adaptable([1,2,23]) #bin aa and bin properties and aminoacid composition

    print(test1)
    print(len(test1))
    print(test1_2)
    print(len(test1_2))
    stop = timeit.default_timer()
    print('Time: ', stop - start)  # 8''/350 aa

    print('tests protein2')
    start = timeit.default_timer()
    test2=protein2.get_all()
    test2_2=protein2.adaptable([1,2,23])
    print(test2)
    print(len(test2))
    print(test2_2)
    print(len(test2_2))
    stop = timeit.default_timer()
    print('Time: ', stop - start) # <4''/60 aa


if __name__=="__main__":
    test_descriptors()
