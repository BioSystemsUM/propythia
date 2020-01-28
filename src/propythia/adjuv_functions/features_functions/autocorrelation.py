# -*- coding: utf-8 -*-
"""
This module is used for computing the Autocorrelation based different

 properties of AADs.You can also input your properties of AADs, then it can help you

to compute Autocorrelation based on the property of AADs. Currently, You

can get 720  for a given protein sequence based on our provided physicochemical

properties of AADs. You can freely use and distribute it. If you hava  any problem,

you could contact with us timely!

References:

[1]: http://www.genome.ad.jp/dbget/aaindex.html

[2]:Feng, Z.P. and Zhang, C.T. (2000) Prediction of membrane protein types based on

the hydrophobic index of amino acids. J Protein Chem, 19, 269-275.

[3]:Horne, D.S. (1988) Prediction of protein helix content from an autocorrelation

analysis of sequence hydrophobicities. Biopolymers, 27, 451-477.

[4]:Sokal, R.R. and Thomson, B.A. (2006) Population structure inferred by local

spatial autocorrelation: an Usage from an Amerindian tribal population. Am J

Phys Anthropol, 129, 121-131.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2010.11.22

Email: oriental-cds@163.com

Altered to python 3.6 for Ana Marta Sequeira 05/2019
"""

import math,string


AALetter=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]


_hydrophobicity = {"A":0.02, "R":-0.42, "N":-0.77, "D":-1.04, "C":0.77, "Q":-1.10, "E":-1.14, "G":-0.80, "H":0.26, "I":1.81, "L":1.14, "K":-0.41, "M":1.00, "F":1.35, "P":-0.09, "S":-0.97, "T":-0.77, "W":1.71, "Y":1.11, "V":1.13}

_av_flexibility = {"A":0.357, "R":0.529, "N":0.463, "D":0.511, "C":0.346, "Q":0.493, "E":0.497, "G":0.544, "H":0.323, "I":0.462, "L":0.365, "K":0.466, "M":0.295, "F":0.314, "P":0.509, "S":0.507, "T":0.444, "W":0.305, "Y":0.420, "V":0.386}

_polarizability = {"A":0.046, "R":0.291, "N":0.134, "D":0.105, "C":0.128, "Q":0.180, "E":0.151, "G":0.000, "H":0.230, "I":0.186, "L":0.186, "K":0.219, "M":0.221, "F":0.290, "P":0.131, "S":0.062, "T":0.108, "W":0.409, "Y":0.298, "V":0.140}

_free_energy = {"A":-0.368, "R":-1.03, "N":0.0, "D":2.06, "C":4.53, "Q":0.731, "E":1.77, "G":-0.525, "H":0.0, "I":0.791, "L":1.07, "K":0.0, "M":0.656, "F":1.06, "P":-2.24, "S":-0.524, "T":0.0, "W":1.60, "Y":4.91, "V":0.401}

_residue_asa = {"A":115.0, "R":225.0, "N":160.0, "D":150.0, "C":135.0, "Q":180.0, "E":190.0, "G":75.0, "H":195.0, "I":175.0, "L":170.0, "K":200.0, "M":185.0, "F":210.0, "P":145.0, "S":115.0, "T":140.0, "W":255.0, "Y":230.0, "V":155.0}

_residue_vol = {"A":52.6, "R":109.1, "N":75.7, "D":68.4, "C":68.3, "Q":89.7, "E":84.7, "G":36.3, "H":91.9, "I":102.0, "L":102.0, "K":105.1, "M":97.7, "F":113.9, "P":73.6, "S":54.9, "T":71.2, "W":135.4, "Y":116.2, "V":85.1}

_steric = {"A":0.52, "R":0.68, "N":0.76, "D":0.76, "C":0.62, "Q":0.68, "E":0.68, "G":0.00, "H":0.70, "I":1.02, "L":0.98, "K":0.68, "M":0.78, "F":0.70, "P":0.36, "S":0.53, "T":0.50, "W":0.70, "Y":0.70, "V":0.76}

_mutability = {"A":100.0, "R":65.0, "N":134.0, "D":106.0, "C":20.0, "Q":93.0, "E":102.0, "G":49.0, "H":66.0, "I":96.0, "L":40.0, "K":-56.0, "M":94.0, "F":41.0, "P":56.0, "S":120.0, "T":97.0, "W":18.0, "Y":41.0, "V":74.0}


###You can continuely add other properties of AADs to compute the pydpi_py3 of protein sequence.


_aa_property=(_hydrophobicity, _av_flexibility, _polarizability, _free_energy, _residue_asa, _residue_vol, _steric, _mutability)

_aa_property_name=('_Hydrophobicity', '_AvFlexibility', '_Polarizability', '_FreeEnergy', '_ResidueASA', '_ResidueVol', '_Steric', '_Mutability')

##################################################################################################

def _mean(listvalue):
	"""
	The mean value of the list data.
	:param listvalue:
	:return:
	"""

	return sum(listvalue)/len(listvalue)
##################################################################################################


def _std(listvalue, ddof=1):
	"""
	The standard deviation of the list data.
	"""
	mean=_mean(listvalue)
	temp=[math.pow(i-mean,2) for i in listvalue]
	res=math.sqrt(sum(temp)/(len(listvalue)-ddof))
	return res
##################################################################################################


def normalize_each_aap(aap):
	"""
	All of the amino acid indices are centralized and standardized before the calculation.
	:param aap: AAP is a dict form containing the properties of 20 amino acids
	:return: result is the a dict form containing the normalized properties of 20 amino acids.
	"""

	result={}
	if len(list(aap.values()))!=20:
		print('You can not input the correct number of properities of Amino acids!')
	else:
		result={}
		for i,j in list(aap.items()):
			result[i]= (j - _mean(list(aap.values()))) / _std(list(aap.values()), ddof=0)

	return result


def calculate_each_normalized_moreau_broto_auto(protein_sequence, aap, aap_name):
	"""
	you can use the function to compute MoreauBrotoAuto for different properties based on AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:param aap: AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	:param aap_name: AAPName is a string used for indicating the property (e.g., '_AvFlexibility')
	:return: result is a dict form containing 30 Normalized Moreau-Broto autocorrelation based on the given property.
	"""
	aa_pdic = normalize_each_aap(aap)

	result={}
	for i in range(1,31):
		temp=0
		for j in range(len(protein_sequence) - i):
			temp= temp + aa_pdic[protein_sequence[j]] * aa_pdic[protein_sequence[j + 1]]
		if len(protein_sequence)-i==0:
			result['MoreauBrotoAuto' + aap_name + str(i)]=round(temp / (len(protein_sequence)), 3)
		else:
			result['MoreauBrotoAuto' + aap_name + str(i)]=round(temp / (len(protein_sequence) - i), 3)

	return result


def calculate_each_moran_auto(protein_sequence, aap, aap_name):
	"""
	you can use the function to compute MoranAuto for different properties based on AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:param aap: AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	:param aap_name: AAPName is a string used for indicating the property (e.g., '_AvFlexibility').
	:return: result is a dict form containing 30 Moran autocorrelation based on the given property.
	"""

	aa_pdic=normalize_each_aap(aap)

	cds=0
	for i in AALetter:
		cds= cds + (protein_sequence.count(i)) * (aa_pdic[i])
	Pmean=cds/len(protein_sequence)

	cc=[]
	for i in protein_sequence:
		cc.append(aa_pdic[i])

	k=(_std(cc,ddof=0))**2

	result={}
	for i in range(1,31):
		temp=0
		for j in range(len(protein_sequence) - i):
				
			temp= temp + (aa_pdic[protein_sequence[j]] - Pmean) * (aa_pdic[protein_sequence[j + i]] - Pmean)
		if len(protein_sequence)-i == 0:
			result['MoranAuto' + aap_name + str(i)]=round(temp / (len(protein_sequence)) / k, 3)
		else:
			result['MoranAuto' + aap_name + str(i)]=round(temp / (len(protein_sequence) - i) / k, 3)

	return result


def calculate_each_geary_auto(protein_sequence, aap, aap_name):
	"""
	you can use the function to compute GearyAuto for different properties based on AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:param aap: AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	:param aap_name: AAPName is a string used for indicating the property (e.g., '_AvFlexibility').
	:return: result is a dict form containing 30 Geary autocorrelation based on the given property.
	"""

	aa_pdic=normalize_each_aap(aap)

	cc=[]
	for i in protein_sequence:
		cc.append(aa_pdic[i])

	K= ((_std(cc))**2) * len(protein_sequence) / (len(protein_sequence) - 1)
	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(protein_sequence) - i):
				
			temp= temp + (aa_pdic[protein_sequence[j]] - aa_pdic[protein_sequence[j + i]]) ** 2
		if len(protein_sequence)-i==0:
			Result['GearyAuto' + aap_name + str(i)]=round(temp / (2 * (len(protein_sequence))) / K, 3)
		else:
			Result['GearyAuto' + aap_name + str(i)]=round(temp / (2 * (len(protein_sequence) - i)) / K, 3)
	return Result
##################################################################################################


def calculate_normalized_moreau_broto_auto(protein_sequence, aa_property, aa_property_name):
	"""
	A method used for computing MoreauBrotoAuto for all properties.

	:param protein_sequence: protein is a pure protein sequence.
	:param aa_property: AAProperty is a list or tuple form containing the properties of 20 amino acids (e.g., _AAProperty).
	:param aa_property_name: AAPName is a list or tuple form used for indicating the property (e.g., '_AAPropertyName').
	:return: result is a dict form containing 30*p Normalized Moreau-Broto autocorrelation based on the given properties.
	"""
	result = {}
	for i in range(len(aa_property)):
		result[aa_property_name[i]]=calculate_each_normalized_moreau_broto_auto(protein_sequence, aa_property[i], aa_property_name[i])
	return result


def calculate_moran_auto(protein_sequence, aa_property, aa_property_name):
	"""
	A method used for computing MoranAuto for all properties
	:param protein_sequence: protein is a pure protein sequence.
	:param aa_property: AAProperty is a list or tuple form containing the properties of 20 amino acids (e.g., _AAProperty).
	:param aa_property_name: AAPName is a list or tuple form used for indicating the property (e.g., '_AAPropertyName').
	:return: result is a dict form containing 30*p Moran autocorrelation based on the given properties.
	"""

	result={}
	for i in range(len(aa_property)):
		result[aa_property_name[i]]=calculate_each_moran_auto(protein_sequence, aa_property[i], aa_property_name[i])

	return result


def calculate_geary_auto(protein_sequence, aa_property, aa_property_name):
	"""
	A method used for computing GearyAuto for all properties
	:param protein_sequence: protein is a pure protein sequence
	:param aa_property: AAProperty is a list or tuple form containing the properties of 20 amino acids (e.g., _AAProperty).
	:param aa_property_name: AAPName is a list or tuple form used for indicating the property (e.g., '_AAPropertyName').
	:return: result is a dict form containing 30*p Geary autocorrelation based on the given properties.
	"""

	result={}
	for i in range(len(aa_property)):
		result[aa_property_name[i]]=calculate_each_geary_auto(protein_sequence, aa_property[i], aa_property_name[i])

	return result

########################NormalizedMoreauBorto##################################


def calculate_normalized_moreau_broto_auto_hydrophobicity(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation based on hydrophobicity.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on Hydrophobicity.
	"""

	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _hydrophobicity, '_Hydrophobicity')
	return result


def calculate_normalized_moreau_broto_auto_av_flexibility(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation based on AvFlexibility.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on AvFlexibility.
	"""
	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _av_flexibility, '_AvFlexibility')
	return result


def calculate_normalized_moreau_broto_auto_polarizability(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation based on Polarizability.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on Polarizability.
	"""

	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _polarizability, '_Polarizability')
	return result


def calculate_normalized_moreau_broto_auto_free_energy(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation  based on FreeEnergy.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on FreeEnergy.
	"""
	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _free_energy, '_FreeEnergy')
	return result



def calculate_normalized_moreau_broto_auto_residue_asa(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation based on ResidueASA.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on ResidueASA.
	"""
	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _residue_asa, '_ResidueASA')
	return result


def calculate_normalized_moreau_broto_auto_residue_vol(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation based on ResidueVol.
	:param protein_sequence: protein is a pure protein sequence.
	:return:  result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on ResidueVol.
	"""
	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _residue_vol, '_ResidueVol')
	return result


def calculate_normalized_moreau_broto_auto_steric(protein_sequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation  based on Steric.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on Steric.
	"""
	result=calculate_each_normalized_moreau_broto_auto(protein_sequence, _steric, '_Steric')
	return result


def CalculateNormalizedMoreauBrotoAutoMutability(ProteinSequence):
	"""
	Calculate the NormalizedMoreauBorto Autocorrelation based on Mutability.
	:param ProteinSequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation based on Mutability.
	"""
	result=calculate_each_normalized_moreau_broto_auto(ProteinSequence, _mutability, '_Mutability')
	return result


##############################MoranAuto######################################

def calculate_moran_auto_hydrophobicity(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation  based on hydrophobicity.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on hydrophobicity.
	"""
	result=calculate_each_moran_auto(protein_sequence, _hydrophobicity, '_Hydrophobicity')
	return result
	

def calculate_moran_auto_av_flexibility(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on	AvFlexibility.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on AvFlexibility.
	"""
	result=calculate_each_moran_auto(protein_sequence, _av_flexibility, '_AvFlexibility')
	return result


def calculate_moran_auto_polarizability(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on Polarizability.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on Polarizability.
	"""
	result=calculate_each_moran_auto(protein_sequence, _polarizability, '_Polarizability')
	return result


def calculate_moran_auto_free_energy(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on FreeEnergy.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on FreeEnergy.
	"""
	result=calculate_each_moran_auto(protein_sequence, _free_energy, '_FreeEnergy')
	return result


def calculate_moran_auto_residue_asa(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on ResidueASA.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on ResidueASA.
	"""
	result=calculate_each_moran_auto(protein_sequence, _residue_asa, '_ResidueASA')
	return result


def calculate_moran_auto_residue_vol(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on ResidueVol.
	:param protein_sequence:  protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on ResidueVol.
	"""
	result=calculate_each_moran_auto(protein_sequence, _residue_vol, '_ResidueVol')
	return result


def calculate_moran_auto_steric(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on AutoSteric.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on AutoSteric.
	"""
	result=calculate_each_moran_auto(protein_sequence, _steric, '_Steric')
	return result


def calculate_moran_auto_mutability(protein_sequence):
	"""
	Calculate the MoranAuto Autocorrelation based on Mutability.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Moran Autocorrelation based on Mutability.
	"""
	result=calculate_each_moran_auto(protein_sequence, _mutability, '_Mutability')
	return result

################################GearyAuto#####################################


def calculate_geary_auto_hydrophobicity(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on hydrophobicity.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on hydrophobicity.
	"""
	result=calculate_each_geary_auto(protein_sequence, _hydrophobicity, '_Hydrophobicity')
	return result
	

def calculate_geary_auto_av_flexibility(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on AvFlexibility.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on AvFlexibility.
	"""
	result=calculate_each_geary_auto(protein_sequence, _av_flexibility, '_AvFlexibility')
	return result


def calculate_geary_auto_polarizability(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on Polarizability.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on Polarizability.
	"""
	result=calculate_each_geary_auto(protein_sequence, _polarizability, '_Polarizability')
	return result


def calculate_geary_auto_free_energy(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation based on FreeEnergy.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on FreeEnergy.
	"""
	result=calculate_each_geary_auto(protein_sequence, _free_energy, '_FreeEnergy')
	return result


def calculate_geary_auto_residue_asa(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on ResidueASA.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on ResidueASA.
	"""
	result = calculate_each_geary_auto(protein_sequence, _residue_asa, '_ResidueASA')
	return result


def calculate_geary_auto_residue_vol(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on ResidueVol.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on ResidueVol.
	"""
	result = calculate_each_geary_auto(protein_sequence, _residue_vol, '_ResidueVol')
	return result


def calculate_geary_auto_steric(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on Steric.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on Steric.
	"""
	result=calculate_each_geary_auto(protein_sequence, _steric, '_Steric')
	return result


def calculate_geary_auto_mutability(protein_sequence):
	"""
	Calculate the GearyAuto Autocorrelation  based on Mutability.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30 Geary Autocorrelation based on Mutability.
	"""
	result=calculate_each_geary_auto(protein_sequence, _mutability, '_Mutability')
	return result
##################################################################################################


def calculate_normalized_moreau_broto_auto_total(protein_sequence):
	"""
	A method used for computing normalized Moreau Broto autocorrelation  based on 8 proterties of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30*8=240 normalized Moreau Broto autocorrelation  based on the given
	properties(i.e., _AAPropert).
	"""
	result={}
	result.update(calculate_normalized_moreau_broto_auto_hydrophobicity(protein_sequence))
	result.update(calculate_normalized_moreau_broto_auto_av_flexibility(protein_sequence))
	result.update(calculate_normalized_moreau_broto_auto_polarizability(protein_sequence))
	result.update(calculate_normalized_moreau_broto_auto_free_energy(protein_sequence))
	result.update(calculate_normalized_moreau_broto_auto_residue_asa(protein_sequence))
	result.update(calculate_normalized_moreau_broto_auto_residue_vol(protein_sequence))
	result.update(calculate_normalized_moreau_broto_auto_steric(protein_sequence))
	result.update(CalculateNormalizedMoreauBrotoAutoMutability(protein_sequence))
	return result

def calculate_moran_auto_total(protein_sequence):
	"""
	A method used for computing Moran autocorrelation based on 8 properties of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30*8=240 Moran autocorrelation  based on the given properties(i.e., _AAPropert).
	"""
	result={}
	result.update(calculate_moran_auto_hydrophobicity(protein_sequence))
	result.update(calculate_moran_auto_av_flexibility(protein_sequence))
	result.update(calculate_moran_auto_polarizability(protein_sequence))
	result.update(calculate_moran_auto_free_energy(protein_sequence))
	result.update(calculate_moran_auto_residue_asa(protein_sequence))
	result.update(calculate_moran_auto_residue_vol(protein_sequence))
	result.update(calculate_moran_auto_steric(protein_sequence))
	result.update(calculate_moran_auto_mutability(protein_sequence))
	return result

def calculate_geary_auto_total(protein_sequence):
	"""
	A method used for computing Geary autocorrelation based on 8 properties of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30*8=240 Geary autocorrelation based on the given
	properties(i.e., _AAPropert).
	"""
	result={}
	result.update(calculate_geary_auto_hydrophobicity(protein_sequence))
	result.update(calculate_geary_auto_av_flexibility(protein_sequence))
	result.update(calculate_geary_auto_polarizability(protein_sequence))
	result.update(calculate_geary_auto_free_energy(protein_sequence))
	result.update(calculate_geary_auto_residue_asa(protein_sequence))
	result.update(calculate_geary_auto_residue_vol(protein_sequence))
	result.update(calculate_geary_auto_steric(protein_sequence))
	result.update(calculate_geary_auto_mutability(protein_sequence))
	return result

##################################################################################################


def calculate_auto_total(protein_sequence):
	"""
	A method used for computing all autocorrelation  based on 8 properties of AADs.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing 30*8*3=720 normalized Moreau Broto, Moran, and Geary
	autocorrelation based on the given properties(i.e., _AAPropert).
	"""
	result={}
	result.update(calculate_normalized_moreau_broto_auto_total(protein_sequence))
	result.update(calculate_moran_auto_total(protein_sequence))
	result.update(calculate_geary_auto_total(protein_sequence))
	return result


if __name__ == "__main__":
	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
	temp1=calculate_normalized_moreau_broto_auto(protein, aa_property=_aa_property, aa_property_name=_aa_property_name)
	#print temp1
	temp2=calculate_moran_auto_mutability(protein)
	print(temp2)
	print(len(calculate_auto_total(protein)))


