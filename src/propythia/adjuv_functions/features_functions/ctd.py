# -*- coding: utf-8 -*-
"""
#####################################################################################

This module is used for computing the composition, transition and distribution 

based on the different properties of AADs. The AADs with the same

properties is marked as the same number. You can get 147  for a given

protein sequence. You can freely use and distribute it. If you hava  any problem, 

you could contact with us timely!

References:

[1]: Inna Dubchak, Ilya Muchink, Stephen R.Holbrook and Sung-Hou Kim. Prediction 

of protein folding class using global description of amino acid sequence. Proc.Natl.

Acad.Sci.USA, 1995, 92, 8700-8704.

[2]:Inna Dubchak, Ilya Muchink, Christopher Mayor, Igor Dralyuk and Sung-Hou Kim. 

Recognition of a Protein Fold in the Context of the SCOP classification. Proteins: 

Structure, Function and Genetics,1999,35,401-407.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2010.11.22

Email: oriental-cds@163.com


Altered and converted to python 3.6 for Ana Marta Sequeira 05/2019
#####################################################################################

"""

import string, math, copy


AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

_Hydrophobicity = {'1':'RKEDQN','2':'GASTPHY','3':'CLVIMFW'}
#'1'stand for Polar; '2'stand for Neutral, '3' stand for Hydrophobicity

_NormalizedVDWV = {'1':'GASTPD','2':'NVEQIL','3':'MHKFRYW'}
#'1'stand for (0-2.78); '2'stand for (2.95-4.0), '3' stand for (4.03-8.08)

_Polarity = {'1':'LIFWCMVY','2':'CPNVEQIL','3':'KMHFRYW'}
#'1'stand for (4.9-6.2); '2'stand for (8.0-9.2), '3' stand for (10.4-13.0)

_Charge = {'1':'KR','2':'ANCQGHILMFPSTWYV','3':'DE'}
#'1'stand for Positive; '2'stand for Neutral, '3' stand for Negative

_SecondaryStr = {'1':'EALMQKRH','2':'VIYCWFT','3':'GNPSD'}
#'1'stand for Helix; '2'stand for Strand, '3' stand for coil

_SolventAccessibility = {'1':'ALFCGIVW','2':'RKQEND','3':'MPSTHY'}
#'1'stand for Buried; '2'stand for Exposed, '3' stand for Intermediate

_Polarizability = {'1':'GASDT','2':'CPNVEQIL','3':'KMHFRYW'}
#'1'stand for (0-0.108); '2'stand for (0.128-0.186), '3' stand for (0.219-0.409)


##You can continuely add other properties of AADs to compute  of protein sequence.

_AATProperty=(_Hydrophobicity,_NormalizedVDWV,_Polarity,_Charge,_SecondaryStr,_SolventAccessibility,_Polarizability)

_AATPropertyName=('_Hydrophobicity','_NormalizedVDWV','_Polarity','_Charge','_SecondaryStr','_SolventAccessibility','_Polarizability')


##################################################################################################


def stringto_num(protein_sequence, aa_property):
	"""
	Tranform the protein sequence into the string form such as 32123223132121123.
	:param protein_sequence:  protein is a pure protein sequence.
	:param aa_property: AAProperty is a dict form containing classifciation of amino acids such as _Polarizability
	:return: result is a string such as 123321222132111123222
	"""

	hard_protein_sequence=copy.deepcopy(protein_sequence)
	for k,m in list(aa_property.items()):
		for index in m:
			hard_protein_sequence=str.replace(hard_protein_sequence,index,k)
	t_protein_sequence=hard_protein_sequence

	return t_protein_sequence


def calculate_composition(protein_sequence, aa_property, aap_name):
	"""
	A method used for computing composition
	:param protein_sequence: protein is a pure protein sequence
	:param aa_property: dict form containing classifciation of amino acids such as _Polarizability.
	:param aap_name: string used for indicating a AAP name.
	:return: dict form containing composition based on the given property.
	"""
	t_protein_sequence=stringto_num(protein_sequence, aa_property)
	result={}
	num=len(t_protein_sequence)
	result[aap_name + 'C' + '1']=round(float(t_protein_sequence.count('1')) / num, 3)
	result[aap_name + 'C' + '2']=round(float(t_protein_sequence.count('2')) / num, 3)
	result[aap_name + 'C' + '3']=round(float(t_protein_sequence.count('3')) / num, 3)
	return result


def calculate_transition(protein_sequence, aa_property, aap_name):
	"""
	A method used for computing transition
	:param protein_sequence: protein is a pure protein sequence
	:param aa_property: AAProperty is a dict form containing classifciation of amino acids such as _Polarizability
	:param aap_name: string used for indicating a AAP name
	:return: dict form containing transition based on the given property
	"""
	t_protein_sequence=stringto_num(protein_sequence, aa_property)
	result={}
	num=len(t_protein_sequence)
	CTD=t_protein_sequence
	result[aap_name + 'T' + '12']=round(float(CTD.count('12') + CTD.count('21')) / (num - 1), 3)
	result[aap_name + 'T' + '13']=round(float(CTD.count('13') + CTD.count('31')) / (num - 1), 3)
	result[aap_name + 'T' + '23']=round(float(CTD.count('23') + CTD.count('32')) / (num - 1), 3)
	return result


def calculate_distribution(protein_sequence, aa_property, aap_name):
	"""
	A method used for computing distribution
	:param protein_sequence: protein is a pure protein sequence
	:param aa_property: dict form containing classification of amino acids such as _Polarizability
	:param aap_name: string used for indicating a AAP name
	:return: dict form containing Distribution based on the given property
	"""
	t_protein_sequence=stringto_num(protein_sequence, aa_property)
	result={}
	num=len(t_protein_sequence)
	temp=('1','2','3')
	for i in temp:
		num=t_protein_sequence.count(i)
		ink=1
		indexk=0
		cds=[]
		while ink<=num:
			indexk=str.find(t_protein_sequence,i,indexk)+1
			cds.append(indexk)
			ink=ink+1
				
		if cds==[]:
			result[aap_name + 'D' + i + '001']=0
			result[aap_name + 'D' + i + '025']=0
			result[aap_name + 'D' + i + '050']=0
			result[aap_name + 'D' + i + '075']=0
			result[aap_name + 'D' + i + '100']=0
		else:
				
			result[aap_name + 'D' + i + '001']=round(float(cds[0]) / num * 100, 3)
			result[aap_name + 'D' + i + '025']=round(float(cds[int(math.floor(num * 0.25)) - 1]) / num * 100, 3)
			result[aap_name + 'D' + i + '050']=round(float(cds[int(math.floor(num * 0.5)) - 1]) / num * 100, 3)
			result[aap_name + 'D' + i + '075']=round(float(cds[int(math.floor(num * 0.75)) - 1]) / num * 100, 3)
			result[aap_name + 'D' + i + '100']=round(float(cds[-1]) / num * 100, 3)

	return result

##################################################################################################


def calculate_composition_hydrophobicity(protein_sequence):
	"""
	A method used for calculating composition based on Hydrophobicity of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Composition based on Hydrophobicity.
	"""
	result=calculate_composition(protein_sequence, _Hydrophobicity, '_Hydrophobicity')
	return result


def calculate_composition_normalized_vdwv(protein_sequence):
	"""
	A method used for calculating composition based on NormalizedVDWV of AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing Composition based on NormalizedVDWV
	"""
	result=calculate_composition(protein_sequence, _NormalizedVDWV, '_NormalizedVDWV')
	return result


def calculate_composition_polarity(protein_sequence):
	"""
	calculating composition based on Polarity of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Composition based on Polarity
	"""
	
	result=calculate_composition(protein_sequence, _Polarity, '_Polarity')
	return result


def calculate_composition_charge(protein_sequence):
	"""
	alculating composition based on Charge of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Composition based on Charge
	"""
	result=calculate_composition(protein_sequence, _Charge, '_Charge')
	return result


def calculate_composition_secondary_str(protein_sequence):
	"""
	 method used for calculating composition based on SecondaryStr of AADs
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Composition based on SecondaryStr
	"""
	result=calculate_composition(protein_sequence, _SecondaryStr, '_SecondaryStr')
	return result


def calculate_composition_solvent_accessibility(protein_sequence):
	"""
	A method used for calculating composition based on SolventAccessibility of  AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Composition based on SolventAccessibility
	"""
	result=calculate_composition(protein_sequence, _SolventAccessibility, '_SolventAccessibility')
	return result


def calculate_composition_polarizability(protein_sequence):
	"""
	A method used for calculating composition based on Polarizability of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: ict form containing Composition based on Polarizability
	"""
	
	result=calculate_composition(protein_sequence, _Polarizability, '_Polarizability')
	return result

##################################################################################################


def calculate_transition_hydrophobicity(protein_sequence):
	"""
	A method used for calculating Transition based on Hydrophobicity of AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing Transition based on Hydrophobicity.
	"""
	result=calculate_transition(protein_sequence, _Hydrophobicity, '_Hydrophobicity')
	return result


def calculate_transition_normalized_vdwv(protein_sequence):
	"""
	A method used for calculating Transition based on NormalizedVDWV of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing Transition based on NormalizedVDWV.
	"""
	result=calculate_transition(protein_sequence, _NormalizedVDWV, '_NormalizedVDWV')
	return result


def calculate_transition_polarity(protein_sequence):
	"""
	A method used for calculating Transition based on Polarity of	AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing Transition based on Polarity.
	"""
	result=calculate_transition(protein_sequence, _Polarity, '_Polarity')
	return result


def calculate_transition_charge(protein_sequence):
	"""
	A method used for calculating Transition based on Charge of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing Transition based on Charge.
	"""
	result=calculate_transition(protein_sequence, _Charge, '_Charge')
	return result


def calculate_transition_secondary_str(protein_sequence):
	"""
	A method used for calculating Transition based on SecondaryStr of	AADs
	:param protein_sequence: protein is a pure protein sequence
	:return: result is a dict form containing Transition based on SecondaryStr
	"""
	result=calculate_transition(protein_sequence, _SecondaryStr, '_SecondaryStr')
	return result


def calculate_transition_solvent_accessibility(protein_sequence):
	"""
	A method used for calculating Transition based on SolventAccessibility of  AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing Transition based on SolventAccessibility.
	"""
	result=calculate_transition(protein_sequence, _SolventAccessibility, '_SolventAccessibility')
	return result


def calculate_transition_polarizability(protein_sequence):
	"""
	A method used for calculating Transition based on Polarizability of AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing Transition based on Polarizability.
	"""
	result=calculate_transition(protein_sequence, _Polarizability, '_Polarizability')
	return result

##################################################################################################


def calculate_distribution_hydrophobicity(protein_sequence):
	"""
	A method used for calculating Distribution based on Hydrophobicity of AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing Distribution based on Hydrophobicity.
	"""
	result=calculate_distribution(protein_sequence, _Hydrophobicity, '_Hydrophobicity')
	return result


def calculate_distribution_normalized_vdwv(protein_sequence):
	"""
	A method used for calculating Distribution based on NormalizedVDWV of	AADs
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Distribution based on NormalizedVDWV
	"""
	result=calculate_distribution(protein_sequence, _NormalizedVDWV, '_NormalizedVDWV')
	return result


def calculate_distribution_polarity(protein_sequence):
	"""
	A method used for calculating Distribution based on Polarity of AADs
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Distribution based on Polarity
	"""
	result=calculate_distribution(protein_sequence, _Polarity, '_Polarity')
	return result


def calculate_distribution_charge(protein_sequence):
	"""
	A method used for calculating Distribution based on Charge of	AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Distribution based on Charge
	"""
	result=calculate_distribution(protein_sequence, _Charge, '_Charge')
	return result
	
def calculate_distribution_secondary_str(protein_sequence):
	"""
	A method used for calculating Distribution based on SecondaryStr of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Distribution based on SecondaryStr.
	"""
	result=calculate_distribution(protein_sequence, _SecondaryStr, '_SecondaryStr')
	return result
	
def calculate_distribution_solvent_accessibility(protein_sequence):
	"""
	A method used for calculating Distribution based on SolventAccessibility of  AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing Distribution based on SolventAccessibility
	"""
	result=calculate_distribution(protein_sequence, _SolventAccessibility, '_SolventAccessibility')
	return result
	
def calculate_distribution_polarizability(protein_sequence):
	"""
	A method used for calculating Distribution based on Polarizability of	AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing Distribution based on Polarizability
	"""
	result=calculate_distribution(protein_sequence, _Polarizability, '_Polarizability')
	return result

##################################################################################################


def calculate_c(protein_sequence):
	"""
	Calculate all composition seven different properties of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: dict form containing all composition
	"""
	result={}
	result.update(calculate_composition_polarizability(protein_sequence))
	result.update(calculate_composition_solvent_accessibility(protein_sequence))
	result.update(calculate_composition_secondary_str(protein_sequence))
	result.update(calculate_composition_charge(protein_sequence))
	result.update(calculate_composition_polarity(protein_sequence))
	result.update(calculate_composition_normalized_vdwv(protein_sequence))
	result.update(calculate_composition_hydrophobicity(protein_sequence))
	return result


def calculate_t(protein_sequence):
	"""
	Calculate all transition based seven different properties of AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing all transition
	"""
	result={}
	result.update(calculate_transition_polarizability(protein_sequence))
	result.update(calculate_transition_solvent_accessibility(protein_sequence))
	result.update(calculate_transition_secondary_str(protein_sequence))
	result.update(calculate_transition_charge(protein_sequence))
	result.update(calculate_transition_polarity(protein_sequence))
	result.update(calculate_transition_normalized_vdwv(protein_sequence))
	result.update(calculate_transition_hydrophobicity(protein_sequence))
	return result
	

def calculate_d(protein_sequence):
	"""
	Calculate all distribution based seven different properties of AADs.
	:param protein_sequence: protein is a pure protein sequence
	:return: dict form containing all distribution
	"""
	result={}
	result.update(calculate_distribution_polarizability(protein_sequence))
	result.update(calculate_distribution_solvent_accessibility(protein_sequence))
	result.update(calculate_distribution_secondary_str(protein_sequence))
	result.update(calculate_distribution_charge(protein_sequence))
	result.update(calculate_distribution_polarity(protein_sequence))
	result.update(calculate_distribution_normalized_vdwv(protein_sequence))
	result.update(calculate_distribution_hydrophobicity(protein_sequence))
	return result


def calculate_ctd(protein_sequence):
	"""
	Calculate all CTD based seven different properties of AADs
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing all C
	"""
	result={}
	result.update(calculate_composition_polarizability(protein_sequence))
	result.update(calculate_composition_solvent_accessibility(protein_sequence))
	result.update(calculate_composition_secondary_str(protein_sequence))
	result.update(calculate_composition_charge(protein_sequence))
	result.update(calculate_composition_polarity(protein_sequence))
	result.update(calculate_composition_normalized_vdwv(protein_sequence))
	result.update(calculate_composition_hydrophobicity(protein_sequence))
	result.update(calculate_transition_polarizability(protein_sequence))
	result.update(calculate_transition_solvent_accessibility(protein_sequence))
	result.update(calculate_transition_secondary_str(protein_sequence))
	result.update(calculate_transition_charge(protein_sequence))
	result.update(calculate_transition_polarity(protein_sequence))
	result.update(calculate_transition_normalized_vdwv(protein_sequence))
	result.update(calculate_transition_hydrophobicity(protein_sequence))
	result.update(calculate_distribution_polarizability(protein_sequence))
	result.update(calculate_distribution_solvent_accessibility(protein_sequence))
	result.update(calculate_distribution_secondary_str(protein_sequence))
	result.update(calculate_distribution_charge(protein_sequence))
	result.update(calculate_distribution_polarity(protein_sequence))
	result.update(calculate_distribution_normalized_vdwv(protein_sequence))
	result.update(calculate_distribution_hydrophobicity(protein_sequence))
	return result


if __name__ == "__main__":
	
#	import scipy,string

#	result=scipy.zeros((268,147))
#	f=file('protein1.txt','r')
#	for i,j in enumerate(f:
#		temp=calculate_ctd(string.strip(j))
#		result[i,:]=temp.values()
#	scipy.savetxt('ResultNCTRER.txt', result, fmt='%15.5f',delimiter='')
#
	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
#	print stringto_num(protein,_Hydrophobicity)
#	print calculate_composition(protein,_Hydrophobicity,'_Hydrophobicity')
#	print calculate_transition(protein,_Hydrophobicity,'_Hydrophobicity')
#	print calculate_distribution(protein,_Hydrophobicity,'_Hydrophobicity')
#	print calculate_distribution_solvent_accessibility(protein)
#	print len(calculate_ctd(protein))
#	print len(calculate_c(protein))
#	print len(calculate_t(protein))
#	print len(calculate_d(protein))
	print(calculate_ctd(protein))


