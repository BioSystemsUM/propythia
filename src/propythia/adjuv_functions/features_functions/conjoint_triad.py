# -*- coding: utf-8 -*-
"""
###############################################################################
This module is used for calculating the conjoint triad features only from the 

protein sequence information. You can get 7*7*7=343 features.You can freely 

use and distribute it. If you hava any problem, you could contact with us timely!

Reference:

Juwen Shen, Jian Zhang, Xiaomin Luo, Weiliang Zhu, Kunqian Yu, Kaixian Chen, 

Yixue Li, Huanliang Jiang. Predicting proten-protein interactions based only 

on sequences inforamtion. PNAS. 2007 (104) 4337-4341.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.09.18

Email: oriental-cds@163.com

Altered and converted to python 3.6 for Ana Marta Sequeira 05/2019

###############################################################################
"""

import string

###############################################################################
AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

#a Dipole scale (Debye): -, Dipole<1.0; +, 1.0<Dipole<2.0; ++, 2.0<Dipole<3.0; +++, Dipole>3.0; +'+'+', Dipole>3.0 with opposite orientation.
#b Volume scale (Å3): -, Volume<50; +, Volume> 50.
#c Cys is separated from class 3 because of its ability to form disulfide bonds. 

_repmat={1:["A",'G','V'],2:['I','L','F','P'],3:['Y','M','T','S'],4:['H','N','Q','W'],5:['R','K'],6:['D','E'],7:['C']}

###############################################################################


def _str2_num(proteinsequence):
	"""
	translate the amino acid letter into the corresponding class based on the given form.
	:param proteinsequence:
	:return:
	"""

	repmat={}
	for i in _repmat:
		for j in _repmat[i]:
			repmat[j]=i
			
	res=proteinsequence
	for i in repmat:
		res=res.replace(i,str(repmat[i]))
	return res


def calculate_conjoint_triad(protein_sequence):
	"""
	Calculate the conjoint triad features from protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing all 343 conjoint triad features
	"""
	res={}
	proteinnum=_str2_num(protein_sequence)
	for i in range(8):
		for j in range(8):
			for k in range(8):
				temp='conjtriad'+str(i)+str(j)+str(k)
				res[temp]=proteinnum.count(temp)
	return res


if __name__ == "__main__":
	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
	print(calculate_conjoint_triad(protein))
	print(len(calculate_conjoint_triad(protein)))
