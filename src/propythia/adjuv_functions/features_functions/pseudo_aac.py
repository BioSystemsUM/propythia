# -*- coding: utf-8 -*-
"""
#########################################################################################

Instead of using the conventional 20-D amino acid composition to represent the sample

of a protein, Prof. Kuo-Chen Chou proposed the pseudo amino acid (PseAA) composition 

in order for inluding the sequence-order information. Based on the concept of Chou's 
 
pseudo amino acid composition, the server PseAA was designed in a flexible way, allowing 
 
users to generate various kinds of pseudo amino acid composition for a given protein
 
sequence by selecting different parameters and their combinations. This module aims at 
 
computing two types of PseAA pydpi_py3: Type I and Type II.
 
You can freely use and distribute it. If you have any problem, you could contact 
 
with us timely.

References:

[1]: Kuo-Chen Chou. Prediction of Protein Cellular Attributes Using Pseudo-Amino Acid 

Composition. PROTEINS: Structure, Function, and Genetics, 2001, 43: 246-255.

[2]: http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/

[3]: http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type2.htm

[4]: Kuo-Chen Chou. Using amphiphilic pseudo amino acid composition to predict enzyme 

subfamily classes. Bioinformatics, 2005,21,10-19.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.9.2

Email: oriental-cds@163.com


The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).

The hydrophilicity values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).

The side-chain mass for each of the 20 amino acids.

CRC Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton, Florida (1985).

R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones, Data for Biochemical Research 3rd ed., 

Clarendon Press Oxford (1986).

Altered and converted to python 3.6 for Ana Marta Sequeira 05/2019


#########################################################################################
"""

import string
import math
#import scipy


AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

_Hydrophobicity={"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,"E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,"M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}

_hydrophilicity={"A":-0.5,"R":3.0,"N":0.2,"D":3.0,"C":-1.0,"Q":0.2,"E":3.0,"G":0.0,"H":-0.5,"I":-1.8,"L":-1.8,"K":3.0,"M":-1.3,"F":-2.5,"P":0.0,"S":0.3,"T":-0.4,"W":-3.4,"Y":-2.3,"V":-1.5}

_residuemass={"A":15.0,"R":101.0,"N":58.0,"D":59.0,"C":47.0,"Q":72.0,"E":73.0,"G":1.000,"H":82.0,"I":57.0,"L":57.0,"K":73.0,"M":75.0,"F":91.0,"P":42.0,"S":31.0,"T":45.0,"W":130.0,"Y":107.0,"V":43.0}

_pK1={"A":2.35,"C":1.71,"D":1.88,"E":2.19,"F":2.58,"G":2.34,"H":1.78,"I":2.32,"K":2.20,"L":2.36,"M":2.28,"N":2.18,"P":1.99,"Q":2.17,"R":2.18,"S":2.21,"T":2.15,"V":2.29,"W":2.38,"Y":2.20}

_pK2={"A":9.87,"C":10.78,"D":9.60,"E":9.67,"F":9.24,"G":9.60,"H":8.97,"I":9.76,"K":8.90,"L":9.60,"M":9.21,"N":9.09,"P":10.6,"Q":9.13,"R":9.09,"S":9.15,"T":9.12,"V":9.74,"W":9.39,"Y":9.11}

_pI={"A":6.11,"C":5.02,"D":2.98,"E":3.08,"F":5.91,"G":6.06,"H":7.64,"I":6.04,"K":9.47,"L":6.04,"M":5.74,"N":10.76,"P":6.30,"Q":5.65,"R":10.76,"S":5.68,"T":5.60,"V":6.02,"W":5.88,"Y":5.63}
#############################################################################################


def _mean(listvalue):
	"""
	########################################################################################
	The mean value of the list data.

	result=_mean(listvalue)
	########################################################################################
	"""
	return sum(listvalue)/len(listvalue)
##############################################################################################
def _std(listvalue,ddof=1):
	"""
	########################################################################################
	The standard deviation of the list data.

	result=_std(listvalue)
	########################################################################################
	"""
	mean=_mean(listvalue)
	temp=[math.pow(i-mean,2) for i in listvalue]
	res=math.sqrt(sum(temp)/(len(listvalue)-ddof))
	return res
##############################################################################################


def normalize_each_aap(aap):
	"""
	All of the amino acid indices are centralized and standardized before the calculation.
	:param aap: dict form containing the properties of 20 amino acids
	:return: dict form containing the normalized properties of 20 amino acids.
	"""
	if len(list(aap.values()))!=20:
		print('You can not input the correct number of properities of Amino acids!')
	else:
		Result={}
		for i,j in list(aap.items()):
			Result[i]= (j - _mean(list(aap.values()))) / _std(list(aap.values()), ddof=0)

		return Result
#############################################################################################
#############################################################################################
##################################Type I pydpi_py3#########################################
####################### Pseudo-Amino Acid Composition ############################
#############################################################################################
#############################################################################################


def _get_correlation_function(Ri='S', Rj='D', aap=[_Hydrophobicity, _hydrophilicity, _residuemass]):
	"""
	Computing the correlation between two given amino acids using the above three properties.
	:param Ri: amino acids
	:param Rj: amino acids
	:param aap:
	:return: correlation value between two amino acids.
	"""

	hydrophobicity=normalize_each_aap(aap[0])
	hydrophilicity=normalize_each_aap(aap[1])
	residuemass=normalize_each_aap(aap[2])
	theta1=math.pow(hydrophobicity[Ri] - hydrophobicity[Rj], 2)
	theta2=math.pow(hydrophilicity[Ri] - hydrophilicity[Rj], 2)
	theta3=math.pow(residuemass[Ri] - residuemass[Rj], 2)
	theta=round((theta1+theta2+theta3)/3.0,3)
	return theta


def _get_sequence_order_correlation_factor(protein_sequence, k=1):
	"""
	Computing the Sequence order correlation factor with gap equal to k based on
	[_Hydrophobicity,_hydrophilicity,_residuemass]
	:param protein_sequence: protein is a pure protein sequence
	:param k: gap
	:return: correlation factor value with the gap equal to k
	"""
	length_sequence=len(protein_sequence)
	res=[]
	for i in range(length_sequence-k):
		aa1=protein_sequence[i]
		aa2=protein_sequence[i + k]
		res.append(_get_correlation_function(aa1, aa2))
	result=round(sum(res)/(length_sequence-k),3)
	return result


def get_aa_composition(protein_sequence):
	"""
	Calculate the composition of Amino acids for a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing the composition of 20 amino acids.
	"""
	length_sequence=len(protein_sequence)
	result={}
	for i in AALetter:
		result[i]=round(float(protein_sequence.count(i)) / length_sequence * 100, 3)
	return result


def _get_pseudo_aac1(protein_sequence, lamda=10, weight=0.05):
	"""
	Computing the first 20 of type I pseudo-amino acid compostion based on [_Hydrophobicity,_hydrophilicity,_residuemass].
	:param protein_sequence:
	:param lamda:
	:param weight:
	:return:
	"""
	rightpart=0.0
	for i in range(lamda):
		rightpart= rightpart + _get_sequence_order_correlation_factor(protein_sequence, k=i + 1)
	aac = get_aa_composition(protein_sequence)
	
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['PAAC'+str(index+1)]=round(aac[i]/temp,3)
	
	return result


def _get_pseudo_aac2(protein_sequence, lamda=10, weight=0.05):
	"""
	Computing the last lamda of type I pseudo-amino acid compostion based on
	[_Hydrophobicity,_hydrophilicity,_residuemass].
	:param protein_sequence:
	:param lamda:
	:param weight:
	:return:
	"""
	rightpart=[]
	for i in range(lamda):
		rightpart.append(_get_sequence_order_correlation_factor(protein_sequence, k=i + 1))
	
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+lamda):
		result['PAAC'+str(index+1)]=round(weight*rightpart[index-20]/temp*100,3)
	
	return result


def _get_pseudo_aac(ProteinSequence, lamda=10, weight=0.05):
	"""
	Computing all of type I pseudo-amino acid compostion based on three given
	properties. Note that the number of PAAC strongly depends on the lamda value. if lamda
	= 20, we can obtain 20+20=40 PAAC. The size of these values depends on the
	choice of lamda and weight simultaneously.
	AAP=[_Hydrophobicity,_hydrophilicity,_residuemass]
	:param ProteinSequence: protein is a pure protein sequence.
	:param lamda: lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15
		Note that (1)lamda should NOT be larger than the length of input protein sequence;
		(2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the
		output of PseAA server is the 20-D amino acid composition.
	:param weight: weight factor is designed for the users to put weight on the additional PseAA components
	with respect to the conventional AA components. The user can select any value within the region from 0.05
	to 0.7 for the weight factor.
	:return: dict form containing calculated 20+lamda PAAC
	"""
	res={}
	res.update(_get_pseudo_aac1(ProteinSequence, lamda=lamda, weight=weight))
	res.update(_get_pseudo_aac2(ProteinSequence, lamda=lamda, weight=weight))
	return res

#############################################################################################
##################################Type II pydpi_py3########################################
###############Amphiphilic Pseudo-Amino Acid Composition pydpi_py3#########################
#############################################################################################
#############################################################################################


def _get_correlation_function_for_apaac(Ri='S', Rj='D', AAP=[_Hydrophobicity, _hydrophilicity]):
	"""
	Computing the correlation between two given amino acids using the above two	properties for APAAC (type II PseAAC).
	:param Ri: amino acids
	:param Rj: amino acids
	:param AAP:
	:return:  correlation value between two amino acids
	"""

	hydrophobicity=normalize_each_aap(AAP[0])
	hydrophilicity=normalize_each_aap(AAP[1])
	theta1=round(hydrophobicity[Ri]*hydrophobicity[Rj],3)
	theta2=round(hydrophilicity[Ri]*hydrophilicity[Rj],3)

	return theta1,theta2


def get_sequence_order_correlation_factor_for_apaac(protein_sequence, k=1):
	"""
	Computing the Sequence order correlation factor with gap equal to k based on [_Hydrophobicity,_hydrophilicity]
	for APAAC (type II PseAAC) .

	:param protein_sequence: protein is a pure protein sequence
	:param k: gap
	:return: correlation factor value with the gap equal to k
	"""
	length_sequence=len(protein_sequence)
	res_hydrophobicity=[]
	reshydrophilicity=[]
	for i in range(length_sequence-k):
		AA1=protein_sequence[i]
		AA2=protein_sequence[i + k]
		temp=_get_correlation_function_for_apaac(AA1, AA2)
		res_hydrophobicity.append(temp[0])
		reshydrophilicity.append(temp[1])
	result=[]
	result.append(round(sum(res_hydrophobicity)/(length_sequence-k),3))
	result.append(round(sum(reshydrophilicity)/(length_sequence-k),3))
	return result


def get_a_pseudo_aac1(protein_sequence, lamda=30, weight=0.5):
	"""
	Computing the first 20 of type II pseudo-amino acid compostion based on	[_Hydrophobicity,_hydrophilicity].
	:param protein_sequence:
	:param lamda:
	:param weight:
	:return:
	"""
	rightpart=0.0
	for i in range(lamda):
		rightpart=rightpart+sum(get_sequence_order_correlation_factor_for_apaac(protein_sequence, k=i + 1))
	AAC=get_aa_composition(protein_sequence)
	
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['APAAC'+str(index+1)]=round(AAC[i]/temp,3)
	
	return result


def get_a_pseudo_aac2(protein_sequence, lamda=30, weight=0.5):
	"""
	Computing the last lamda of type II pseudo-amino acid compostion based on	[_Hydrophobicity,_hydrophilicity].
	:param protein_sequence:
	:param lamda:
	:param weight:
	:return:
	"""
	rightpart=[]
	for i in range(lamda):
		temp=get_sequence_order_correlation_factor_for_apaac(protein_sequence, k=i + 1)
		rightpart.append(temp[0])
		rightpart.append(temp[1])
		
	
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+2*lamda):
		result['PAAC'+str(index+1)]=round(weight*rightpart[index-20]/temp*100,3)
	
	return result
	

def get_a_pseudo_aac(protein_sequence, lamda=30, weight=0.5):
	"""
	Computing all of type II pseudo-amino acid compostion based on the given
	properties. Note that the number of PAAC strongly depends on the lamda value. if lamda
	= 20, we can obtain 20+20=40 PAAC. The size of these values depends on the
	choice of lamda and weight simultaneously.
	:param protein_sequence: protein is a pure protein sequence
	:param lamda: lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
		Note that (1)lamda should NOT be larger than the length of input protein sequence;
		(2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the
		output of PseAA server is the 20-D amino acid composition.
	:param weight: weight factor is designed for the users to put weight on the additional PseAA components
	with respect to the conventional AA components. The user can select any value within the
	region from 0.05 to 0.7 for the weight factor.
	:return: dict form containing calculated 20+lamda PAAC
	"""
	res={}
	res.update(get_a_pseudo_aac1(protein_sequence, lamda=lamda, weight=weight))
	res.update(get_a_pseudo_aac2(protein_sequence, lamda=lamda, weight=weight))
	return res


#############################################################################################
#############################################################################################
##################################Type I pydpi_py3#########################################
####################### Pseudo-Amino Acid Composition pydpi_py3############################
#############################based on different properties###################################
#############################################################################################
#############################################################################################
def get_correlation_function(Ri='S', Rj='D',AAP=[]):
	"""
	Computing the correlation between two given amino acids using the given properties.
	:param Ri: amino acids
	:param Rj: amino acids
	:param AAP: list form containing the properties, each of which is a dict form
	:return: correlation value between two amino acids
	"""

	AAP= [_Hydrophobicity,_hydrophilicity,_residuemass,_pK1,_pK2,_pI]
	NumAAP=len(AAP)
	theta=0.0
	for i in range(NumAAP):
		temp=normalize_each_aap(AAP[i])
		theta=theta+math.pow(temp[Ri]-temp[Rj],2)
	result=round(theta/NumAAP,3)
	return result


def get_sequence_order_correlation_factor(protein_sequence, k=1, AAP=[]):
	"""
	Computing the Sequence order correlation factor with gap equal to k based on
	the given properities.
	:param protein_sequence: protein is a pure protein sequence
	:param k: gap
	:param AAP: AAP is a list form containing the properties, each of which is a dict form
	:return: correlation factor value with the gap equal to k
	"""

	length_sequence=len(protein_sequence)
	res=[]
	for i in range(length_sequence-k):
		aa1=protein_sequence[i]
		aa2=protein_sequence[i + k]
		res.append(get_correlation_function(aa1, aa2, AAP))
	result=round(sum(res)/(length_sequence-k),3)
	return result


def get_pseudo_aac1(protein_sequence, lamda=30, weight=0.05, AAP=[]):
	"""
	Computing the first 20 of type I pseudo-amino acid compostion based on the given properties.
	:param protein_sequence:
	:param lamda:
	:param weight:
	:param AAP:
	:return:
	"""
	rightpart=0.0
	for i in range(lamda):
		rightpart= rightpart + get_sequence_order_correlation_factor(protein_sequence, i + 1, AAP)
	aac=get_aa_composition(protein_sequence)
	
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['PAAC'+str(index+1)]=round(aac[i]/temp,3)
	
	return result


def get_pseudo_aac2(protein_sequence, lamda=30, weight=0.05, AAP=[]):
	"""
	Computing the last lamda of type I pseudo-amino acid compostion based on the given properties.
	:param protein_sequence:
	:param lamda:
	:param weight:
	:param AAP:
	:return:
	"""
	rightpart=[]
	for i in range(lamda):
		rightpart.append(get_sequence_order_correlation_factor(protein_sequence, i + 1, AAP))
	
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+lamda):
		result['PAAC'+str(index+1)]=round(weight*rightpart[index-20]/temp*100,3)
	
	return result


def get_pseudo_aac(protein_sequence, lamda=30, weight=0.05, AAP=[]):
	"""
	Computing all of type I pseudo-amino acid compostion based on the given
	properties. Note that the number of PAAC strongly depends on the lamda value. if lamda
	= 20, we can obtain 20+20=40 PAAC. The size of these values depends on the
	choice of lamda and weight simultaneously. You must specify some properties into AAP.
	:param protein_sequence: protein is a pure protein sequence
	:param lamda: 	lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
		Note that (1)lamda should NOT be larger than the length of input protein sequence;
		(2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the
		output of PseAA server is the 20-D amino acid composition.
	:param weight: weight factor is designed for the users to put weight on the additional PseAA components
	with respect to the conventional AA components. The user can select any value within the
	region from 0.05 to 0.7 for the weight factor.
	:param AAP: AAP is a list form containing the properties, each of which is a dict form
	:return: dict form containing calculated 20+lamda PAAC
	"""
	res={}
	res.update(get_pseudo_aac1(protein_sequence, lamda, weight, AAP))
	res.update(get_pseudo_aac2(protein_sequence, lamda, weight, AAP))
	return res


if __name__ == "__main__":
	import string
	protein="MTDRARLRLHDTAAGVVRDFVPLRPGHVSIYLCGATVQGLPHIGHVRSGVAFDILRRWLL\
ARGYDVAFIRNVTDIEDKILAKAAAAGRPWWEWAATHERAFTAAYDALDVLPPSAEPRAT\
GHITQMIEMIERLIQAGHAYTGGGDVYFDVLSYPEYGQLSGHKIDDVHQGEGVAAGKRDQ\
RDFTLWKGEKPGEPSWPTPWGRGRPGWHLECSAMARSYLGPEFDIHCGGMDLVFPHHENE\
IAQSRAAGDGFARYWLHNGWVTMGGEKMSKSLGNVLSMPAMLQRVRPAELRYYLGSAHYR\
SMLEFSETAMQDAVKAYVGLEDFLHRVRTRVGAVCPGDPTPRFAEALDDDLSVPIALAEI\
HHVRAEGNRALDAGDHDGALRSASAIRAMMGILGCDPLDQRWESRDETSAALAAVDVLVQ\
AELQNREKAREQRNWALADEIRGRLKRAGIEVTDTADGPQWSLLGGDTK"
	protein=str.strip(protein)
	PAAC=get_pseudo_aac(protein, lamda=5, AAP=[])

	for i in PAAC:
		print(i, PAAC[i])

	#	temp=_get_correlation_function('S','D')
#	print temp
#	
#	print _get_sequence_order_correlation_factor(protein,k=4)
#	
#	PAAC1=_get_pseudo_aac1(protein,lamda=4)
#	for i in PAAC1:
#		print i, PAAC1[i]
#	PAAC2=_get_pseudo_aac2(protein,lamda=4)
#	for i in PAAC2:
#		print i, PAAC2[i]
#	print len(PAAC1)
#	print _GetSequenceOrderCorrelationFactorForAPAAC(protein,k=1)
#	APAAC1=_GetAPseudoAAC1(protein,lamda=4)
#	for i in APAAC1:
#		print i, APAAC1[i]

#	APAAC2=get_a_pseudo_aac2(protein,lamda=4)
#	for i in APAAC2:
#		print i, APAAC2[i]
#	APAAC=get_a_pseudo_aac(protein,lamda=4)
#	
#	for i in APAAC:
#		print i, APAAC[i]

	# PAAC=get_pseudo_aac(protein,lamda=5,AAP=[_Hydrophobicity,_hydrophilicity])
	#
	# for i in PAAC:
	# 	print(i, PAAC[i])

	protein="AAGMGFFGAR"
	protein=str.strip(protein)
	PAAC=get_pseudo_aac(protein, lamda=5, AAP=[])

	for i in PAAC:
		print(i, PAAC[i])

# [_Hydrophobicity,_hydrophilicity,_residuemass,_pK1,_pK2,_pI]