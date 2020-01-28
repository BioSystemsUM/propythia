# -*- coding: utf-8 -*-
"""
#####################################################################################

Allows to change the sequence and generating other.cut or add aminoacids to obtain sequences with equal lenght.
(one sequence ----> one sequence).
	The function receives a sequence, the desired number of aa from the n terminal and from the c terminal.
	It receives a third argument that is called terminal. This argument is used, to decide in which direction add dummy aa
	(if needed)

	To considerer a protein only from the n terminal, c terminal=0 and the proteins will be cutted from left right.
	If necessary dummy will be added to the right

	To considerer a protein only from the c terminal, n terminal= 0 and the proteins will be cutted from right left.
	If necessary dummy will be added to the left

	To consider both extremes of sequence both n and c terminal are different of zero.
	If len of protein sequence is smaller than the both extremes together, the middle aa will be repeated.
	If necessary dummy aa will be added to the right (terminal=0), to the left (terminal=-1) or in the middle (terminal=2).


Authors:

Date:

Email:

#####################################################################################

"""

import re
import string

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
#############################################################################################


def seq_equal_lenght(seq,n_terminal=10, c_terminal=10, terminal=0):
    """
    cut or add aminoacids to obtain sequences with equal lenght.
    :param seq: protein sequence
    :param n_terminal: number of aa to consider in the n terminal (left side of sequence)
    :param c_terminal: number of aa to consider in the c terminal (right side of sequence)
    :param terminal: in case of the need to add dummy aa and no terminal as already been chosen, it decides where to add
                    0 to add to the right (consider N terminal)
                    -1 to add to the left (consider C terminal)
                    2 to add in the middle (N and C terminal will be both present and repeated with dummy in middle
    :return: list of sequences cntaining all the same lenght
    """

    size=len(seq)
    lenght=n_terminal+c_terminal
    if size>lenght:
        if c_terminal!=0:
            seq=seq[:n_terminal]+seq[-c_terminal:]

        else:
            seq=seq[:n_terminal]

    else: #the size of sequence is less than the lenght desired and dummy aa is needed

        #if one of the terminals is 0 the user already set in each direction to add

        if c_terminal==0:
            seq=seq[:lenght] + str('Z'*(lenght-size))

        elif n_terminal==0:
            seq= str('Z'*(lenght-size)) +seq[-lenght:]

        else: #if both terminals are filled, the variable terminal decide in which direction to cut/add

            if terminal==0: #add dummy to the right
                seq=seq[:lenght] + str('Z'*(lenght-size))


            elif terminal==-1: #add dummy to the left
                seq= str('Z'*(lenght-size)) +seq[-lenght:]


            else:#duplicate aa in the middle retaining the terminals. If necessary add dummy in the middle
                if n_terminal<size and c_terminal<size:
                    seq=seq[:n_terminal]+seq[-c_terminal:]

                elif n_terminal<size and c_terminal>size:
                    seq=seq[:n_terminal]+str('Z'*(c_terminal-size))+seq[-c_terminal:]

                elif n_terminal>size and c_terminal<size:
                    seq=seq[:n_terminal]+str('Z'*(n_terminal-size))+seq[-c_terminal:]

                else:
                    seq=seq[:n_terminal]+str('Z'*(n_terminal-size))+str('Z'*(c_terminal-size))+seq[-c_terminal:]

    return seq


if __name__ == "__main__":
    print((seq_equal_lenght('AAVFNDRAT', 5,0)))
    print((seq_equal_lenght('AAVFNDRAT', 0,5)))
    print((seq_equal_lenght('AAVFNDRAT', 5,5)))

    print((seq_equal_lenght('AAVFNDRAT', 5,5,-1)))
    print((seq_equal_lenght('AAVFNDRAT', 10,10,2)))

    print((seq_equal_lenght('ATN', 5,0)))
    print(seq_equal_lenght('ATN', 0,5))
    print(seq_equal_lenght('ATN', 5,5,2))
    print(seq_equal_lenght('ATN', 5,5,-1))

    print((seq_equal_lenght('AAVFNDRAT', 15,5,2)))





