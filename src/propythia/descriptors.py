# -*- coding: utf-8 -*-
"""
##############################################################################

A class  used for computing different types of protein descriptors.
It contains descriptors from packages pydpi, biopython, pfeature and modlamp.

Authors: Ana Marta Sequeira

Date: 05/2019

Email:

##############################################################################
"""
from propythia.sequence import ReadSequence
from propythia.adjuv_functions.features_functions.aa_index import get_aa_index1, get_aa_index23
from propythia.adjuv_functions.features_functions.binary import bin_aa_ct
from propythia.adjuv_functions.features_functions.binary_aa_properties import bin_pc_wp
from propythia.adjuv_functions.features_functions.descriptors_modlamp import GlobalDescriptor, \
    PeptideDescriptor
from propythia.adjuv_functions.features_functions.bondcomp import boc_wp
from propythia.adjuv_functions.features_functions.aa_composition import calculate_aa_composition, \
    calculate_dipeptide_composition, get_spectrum_dict
from propythia.adjuv_functions.features_functions.pseudo_aac import get_pseudo_aac, get_a_pseudo_aac
from propythia.adjuv_functions.features_functions.autocorrelation import \
    calculate_normalized_moreau_broto_auto_total, calculate_moran_auto_total, calculate_geary_auto_total
from propythia.adjuv_functions.features_functions.ctd import calculate_ctd
from propythia.adjuv_functions.features_functions.quasi_sequence_order import \
    get_sequence_order_coupling_number_total, get_quasi_sequence_order
from propythia.adjuv_functions.features_functions.quasi_sequence_order import \
    get_sequence_order_coupling_numberp, get_quasi_sequence_orderp
from propythia.adjuv_functions.features_functions.conjoint_triad import calculate_conjoint_triad
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import sys
import pandas as pd


class Descriptor:
    """
	The Descriptor class collects all descriptor calculation functions into a simple class.
	Some of the descriptors functions are based on code from pydpi (altered to python3), modlamp, pfature and biopython.
	It returns the features in a dictionary object
	"""

    AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    Version = 1.0

    def __init__(self, protein_sequence):  # the function takes as input the protein sequence
        """	constructor """
        self.ProteinSequence = protein_sequence

    # ################# GET AA INDEX (from specific property) ##################

    def get_aa_index1(self, name, path='.'):
        """
		Get the amino acid property values from aaindex1
		(function from pydpi)
		:param name: name is the name of amino acid property (e.g., KRIW790103)
		:param path: path to get aa index. by default is the one in the package.
		:return: result is a dict form containing the properties of 20 amino acids
		"""

        return get_aa_index1(name, path=path)

    def get_aa_index23(self, name, path='.'):
        """
		Get the amino acid property values from aaindex2 and aaindex3
		(function from from pydpi)
		:param name: name is the name of amino acid property (e.g.,TANS760101,GRAR740104)
		:param path: path to get aa index. by default is the one in the package.
		:return: result is a dict form containing the properties of 400 amino acid pairs
		"""
        return get_aa_index23(name, path=path)

    # ################# BINARY PROFILES DESCRIPTORS  ##################

    def get_bin_aa(self):
        """
		 binary profile of aminoacid composition
		:return: dictionary containing binary profile
		"""

        res = {}
        result = bin_aa_ct(self.ProteinSequence)
        for i in range(len(self.ProteinSequence)):
            for residue in range(len(result) - 1):
                for value in range(20):
                    name_feature = 'bin_aa' + '_' + str(residue) + '_' + str(value)
                    res[name_feature] = result[residue].split(",")[value]

        return res

    def get_bin_resi_prop(self,
                          list_descriptors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                            21, 22, 23, 24]):
        """
		Binary profile of residues for 25 phychem feature
		:param list_descriptors: ist containing feature numbers (valid number, 0-24) (e.g. [5,4,8,24]
		    				FEATURE NAME                    FEATURE NUMBER

							'Positively charged' --                  0
							'Negatively charged' --                  1
							'Neutral charged' --                     2
							'Polarity' --                            3
							'Non polarity' --                        4
							'Aliphaticity' --                        5
							'Cyclic' --                              6
							'Aromaticity' --                         7
							'Acidicity'--                            8
							'Basicity'--                             9
							'Neutral (ph)' --                       10
							'Hydrophobicity' --                     11
							'Hydrophilicity' --                     12
							'Neutral' --                            13
							'Hydroxylic' --                         14
							'Sulphur content' -                     15
							'Secondary Structure(Helix)'            16
							'Secondary Structure(Strands)',         17
							'Secondary Structure(Coil)',            18
							'Solvent Accessibilty (Buried)',        19
							'Solvent Accesibilty(Exposed)',         20
							'Solvent Accesibilty(Intermediate)',    21
							'Tiny',                                 22
							'Small',                                23
							'Large'                                 24
		:return: dict form with binary values
		"""

        res = bin_pc_wp(self.ProteinSequence, list_descriptors)
        return res

    # ################# PHYSICO CHEMICAL DESCRIPTORS  ##################

    def get_lenght(self):
        """
		Calculates lenght of sequence (number of aa)
		:return: dictionary with the value of lenght
		"""
        res = {}
        res['lenght'] = float(len(self.ProteinSequence.strip()))
        return res

    def get_charge(self, ph=7.4, amide=False):
        """
		Calculates charge of sequence (1 value) from modlamp
		:param ph: ph considered to calculate. 7.4 by default
		:param amide: by default is not considered an amide protein sequence.
		:return: dictionary with the value of charge
		"""

        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.calculate_charge(ph=ph, amide=amide)
        res['charge'] = desc.descriptor[0][0]
        return res

    def get_charge_density(self, ph=7.0, amide=False):
        """
		Calculates charge density of sequence (1 value) from modlamp
		:param ph: ph considered to calculate. 7 by default
		:param amide: by default is not considered an amide protein sequence.
		:return: dictionary with the value of charge density
		"""

        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.charge_density(ph, amide)
        res['chargedensity'] = desc.descriptor[0][0]
        return res

    def get_formula(self, amide=False):
        """
		Calculates number of C,H,N,O and S of the aa of sequence (5 values) from modlamp
		:param amide: by default is not considered an amide protein sequence.
		:return: dictionary with the 5 values of C,H,N,O and S
		"""
        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.formula(amide)
        formula = desc.descriptor[0][0].split()
        for atom in formula:
            if atom[0] == 'C': res['formulaC'] = int(atom[1:])
            if atom[0] == 'H': res['formulaH'] = int(atom[1:])
            if atom[0] == 'N': res['formulaN'] = int(atom[1:])
            if atom[0] == 'O': res['formulaO'] = int(atom[1:])
            if atom[0] == 'S': res['formulaS'] = int(atom[1:])
        # some formulas, specially S sometimes culd be a zero, to not transform iinto a nan in dataset
        if not res.get('formulaC'): res['formulaC'] = 0
        if not res.get('formulaH'): res['formulaH'] = 0
        if not res.get('formulaN'): res['formulaN'] = 0
        if not res.get('formulaO'): res['formulaO'] = 0
        if not res.get('formulaS'): res['formulaS'] = 0
        return res

    def get_bond(self):
        """
		This function gives the sum of the bond composition for each type of bond
		For bond composition four types of bonds are considered
		total number of bonds (including aromatic), hydrogen bond, single bond and double bond.
		:return: dictionary with 4 values
		"""
        res = boc_wp(self.ProteinSequence)
        return res

    def get_mw(self):
        """
		Calculates molecular weight of sequence (1 value) from modlamp
		:return: dictionary with the value of molecular weight
		"""

        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.calculate_MW(amide=True)
        res['MW_modlamp'] = desc.descriptor[0][0]
        return res

    # def GetMW2(self): give values different molecular weight from modlamp and biopython.
    # 	"""
    # 	Calculates molecular weight from sequence (1 value) from biopython
    # 	Input:
    # 	Output: dictionary with the value of MW
    # 	"""
    #
    # 	res={}
    # 	analysed_seq = ProteinAnalysis(self.protein_sequence)
    # 	res['MW']= analysed_seq.molecular_weight()
    # 	return res

    def get_gravy(self):
        """
		Calculates Gravy from sequence (1 value) from biopython
		:return: dictionary with the value of gravy
		"""

        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        res['Gravy'] = analysed_seq.gravy()
        return res

    def get_aromacity(self):
        """
		Calculates Aromacity from sequence (1 value) from biopython
		:return: dictionary with the value of aromacity
		"""

        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        res['Aromacity'] = analysed_seq.aromaticity()
        return res

    def get_isoelectric_point(self):
        """
		Calculates Isolectric Point from sequence (1 value) from biopython
		:return: dictionary with the value of Isolectric Point
		"""

        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        res['IsoelectricPoint'] = analysed_seq.isoelectric_point()
        return res

    # def GetIsoelectricPoint_2(self):#give values different molecular weight from modlamp and biopython
    # 	"""
    # 	Calculates Isolectric Point from sequence (1 value) from modlamp
    # 	Input:
    # 	Output: dictionary with the value of Isolectric Point
    # 	"""
    #
    # 	res={}
    # 	desc = GlobalDescriptor(self.protein_sequence)
    # 	desc.isoelectric_point()
    # 	res['isoelectricpoint_modlamp']= desc.descriptor[0][0]
    # 	return res

    def get_instability_index(self):
        """
		Calculates Instability index from sequence (1 value) from biopython
		:return: dictionary with the value of Instability index
		"""
        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        res['Instability_index'] = analysed_seq.instability_index()
        return res

    def get_sec_struct(self):
        """
		Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython
		:return: dictionary with the 3 value of helix, turn, sheet
		"""

        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        res['SecStruct_helix'] = analysed_seq.secondary_structure_fraction()[0]  # helix
        res['SecStruct_turn'] = analysed_seq.secondary_structure_fraction()[1]  # turn
        res['SecStruct_sheet'] = analysed_seq.secondary_structure_fraction()[2]  # sheet
        return res

    def get_molar_extinction_coefficient(
            self):  # [reduced, oxidized] # with reduced cysteines / # with disulfid bridges
        """
		Calculates the molar extinction coefficient (2 values) from biopython
		:return: dictionary with the value of reduced cysteins and oxidized (with disulfid bridges)
		"""

        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        res['Molar_extinction_coefficient_reduced'] = analysed_seq.molar_extinction_coefficient()[0]  # reduced
        res['Molar_extinction_coefficient_oxidized'] = analysed_seq.molar_extinction_coefficient()[1]  # cys cys bounds
        return res

    def get_flexibility(self):
        """
		Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython
		:return: dictionary with proteinsequencelenght-9 values of flexiblity
		"""

        res = {}
        analysed_seq = ProteinAnalysis(self.ProteinSequence)
        flexibility = analysed_seq.flexibility()
        for i in range(len(flexibility)):
            res['flexibility_' + str(i)] = flexibility[i]
        return res

    def get_aliphatic_index(self):
        """
		Calculates aliphatic index of sequence (1 value) from modlamp
		:return: dictionary with the value of aliphatic index
		"""

        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.aliphatic_index()
        res['aliphatic_index'] = desc.descriptor[0][0]
        return res

    def get_boman_index(self):
        """
		Calculates boman index of sequence (1 value) from modlamp
		:return: dictionary with the value of boman index
		"""

        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.boman_index()
        res['bomanindex'] = desc.descriptor[0][0]
        return res

    def get_hydrophobic_ratio(self):
        """
		Calculates hydrophobic ratio of sequence (1 value) from modlamp
		:return: dictionary with the value of hydrophobic ratio
		"""

        res = {}
        desc = GlobalDescriptor(self.ProteinSequence)
        desc.hydrophobic_ratio()
        res['hydrophobic_ratio'] = desc.descriptor[0][0]
        return res

    ################## AMINO ACID COMPOSITION ##################

    def get_aa_comp(self):
        """
		Calculates amino acid compositon (20 values)  from pydpi
		:return: dictionary with the fractions of all 20 aa(keys are the aa)
		"""
        res = calculate_aa_composition(self.ProteinSequence)
        return res

    def get_dp_comp(self):
        """
		Calculates dipeptide composition (400 values) from pydpi
		:return: dictionary with the fractions of all 400 possible combiinations of 2 aa
		"""
        res = calculate_dipeptide_composition(self.ProteinSequence)
        return res

    def get_tp_comp(self):
        """
		Calculates tripeptide composition (8000 values) from pydpi
		:return: dictionary with the fractions of all 8000 possible combinations of 3 aa
		"""

        res = get_spectrum_dict(self.ProteinSequence)
        return res

    ################## PSEUDO AMINO ACID COMPOSITION ##################

    def get_paac(self, lamda=10, weight=0.05):
        """
		Calculates Type I Pseudo amino acid composition (default is 30, depends on lamda) from pydpi
		:param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
					should NOT be larger than the length of input protein sequence
					when lamda =0, the output of PseAA server is the 20-D amino acid composition
		:param weight: weight on the additional PseAA components. with respect to the conventional AA components.
					The user can select any value within the region from 0.05 to 0.7 for the weight factor.
		:return: dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
		"""
        res = get_pseudo_aac(self.ProteinSequence, lamda=lamda, weight=weight)
        return res

    def get_paac_p(self, lamda=10, weight=0.05, AAP=[]):
        """
		Calculates Type I Pseudo amino acid composition for a given property (default is 30, depends on lamda) from pydpi
		:param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
					should NOT be larger than the length of input protein sequence
					when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
		:param weight: weight on the additional PseAA components. with respect to the conventional AA components.
					The user can select any value within the region from 0.05 to 0.7 for the weight factor.
		:param AAP: list of properties. each of which is a dict form.
				PseudoAAC._Hydrophobicity,PseudoAAC._hydrophilicity, PseudoAAC._residuemass,PseudoAAC._pK1,PseudoAAC._pK2,PseudoAAC._pI
		:return: dictionary with the fractions of all PAAC(keys are the PAAC). Number of keys depends on lamda
		"""
        res = get_pseudo_aac(self.ProteinSequence, lamda=lamda, weight=weight, AAP=AAP)
        return res

    def get_apaac(self, lamda=10, weight=0.5):
        """
		Calculates Type II Pseudo amino acid composition - Amphiphilic (default is 30, depends on lamda) from pydpi
		:param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
					should NOT be larger than the length of input protein sequence
					when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
		:param weight: weight on the additional PseAA components. with respect to the conventional AA components.
					The user can select any value within the region from 0.05 to 0.7 for the weight factor.
		:return: dictionary with the fractions of all PAAC(keys are the PAAC). Number of keys depends on lamda
		"""
        res = get_a_pseudo_aac(self.ProteinSequence, lamda=lamda, weight=weight)
        return res

    # ################# AUTOCORRELATION DESCRIPTORS ##################

    def get_moreau_broto_auto(self):
        """
		Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi
		:return: dictionary with the 240 descriptors
		"""
        res = calculate_normalized_moreau_broto_auto_total(self.ProteinSequence)
        return res

    def get_moran_auto(self):
        """
		Calculates  Moran autocorrelation (240 values) from pydpi
		:return: dictionary with the 240 descriptors
		"""
        res = calculate_moran_auto_total(self.ProteinSequence)
        return res

    def get_geary_auto(self):
        """
		Calculates  Geary autocorrelation (240 values) from pydpi
		:return: dictionary with the 240 descriptors
		"""
        res = calculate_geary_auto_total(self.ProteinSequence)
        return res

    # ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

    def get_ctd(self):
        """
		Calculates the Composition Transition Distribution descriptors (147 values) from pydpi
		:return: dictionary with the 147 descriptors
		"""
        res = calculate_ctd(self.ProteinSequence)
        return res

    # ################# CONJOINT TRIAD ##################

    def get_conj_t(self):
        """
		Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi
		:return: dictionary with the 343 descriptors
		"""
        res = calculate_conjoint_triad(self.ProteinSequence)
        return res

    # #################  SEQUENCE ORDER  ##################

    def get_socn(self, maxlag=45):
        """
		Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi
		:param maxlag: maximum lag. Smaller than length of the protein
		:return: dictionary with the descriptors (90 descriptors)
		"""
        res = get_sequence_order_coupling_number_total(self.ProteinSequence, maxlag=maxlag)
        return res

    def get_socn_p(self, maxlag=45, distancematrix={}):
        """
		Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi
		:param maxlag: maximum lag. Smaller than length of the protein
		:param distancematrix: dict form containing 400 distance values
		:return: dictionary with the descriptors (90 descriptors)
		"""

        res = get_sequence_order_coupling_numberp(self.ProteinSequence, maxlag=maxlag, distancematrix=distancematrix)
        return res

    def get_qso(self, maxlag=30, weight=0.1):
        """
		Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi
		:param maxlag: maximum lag. Smaller than length of the protein
		:param weight:
		:return: dictionary with the descriptors (100 descriptors)
		"""
        res = get_quasi_sequence_order(self.ProteinSequence, maxlag=maxlag, weight=weight)
        return res

    def get_qso_p(self, maxlag=30, weight=0.1, distancematrix={}):
        """
		Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi
		:param maxlag: maximum lag. Smaller than length of the protein
		:param weight:
		:param distancematrix: dict form containing 400 distance values
		:return: dictionary with the descriptors (100 descriptors)
		"""

        res = get_quasi_sequence_orderp(self.ProteinSequence, maxlag=maxlag, weight=weight,
                                        distancematrix=distancematrix)
        return res

    # ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

    """amino acid descriptor scales available are the ones from modlamo. 
	For more information please check:  https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor
	amino acid sclaes include AASI, argos, bulkiness, charge_phys, charge_acid, eisenberg and others."""

    def calculate_moment(self, window=1000, angle=100, modality='max', scalename='Eisenberg'):
        """
		Calculates moment of sequence (1 value) from modlamp
		:param window: amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken
		:param angle: angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets
		:param modality: maximum or mean hydrophobic moment
		:param scalename:
		:return: dictionary with one value of moment
		"""

        res = {}
        AMP = PeptideDescriptor(self.ProteinSequence, scalename)
        AMP.calculate_moment(window, angle, modality)
        res['moment'] = AMP.descriptor[0][0]
        return res

    def calculate_global(self, window=1000, modality='max', scalename='Eisenberg'):
        """
		Calculates a global / window averaging descriptor value of a given AA scale of sequence (1 value) from modlamp
		:param window: amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
		:param modality: maximum or mean hydrophobic moment
		:param scalename:
		:return: dictionary with one value
		"""

        res = {}
        AMP = PeptideDescriptor(self.ProteinSequence, scalename)
        AMP.calculate_global(window, modality)
        res['global'] = AMP.descriptor[0][0]
        return res

    def calculate_profile(self, prof_type='uH', window=7, scalename='Eisenberg'):
        """
		Calculates hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercep
		(2 values) from modlamp
		:param prof_type: prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment
		:param window: size of sliding window used (odd-numbered)
		:param scalename:
		:return: dictionary with two value
		"""

        res = {}
        AMP = PeptideDescriptor(self.ProteinSequence, scalename)
        AMP.calculate_profile(prof_type, window)
        desc = AMP.descriptor[0]
        for i in range(len(desc)):
            res['profile_' + str(i)] = desc[i]
        return res

    def calculate_arc(self, modality="max", scalename='peparc'):
        """
		Calculates arcs as seen in the helical wheel plot. Use for binary amino acid scales only ( 5 values) from modlamp
		:param modality: maximum or mean
		:param scalename: binary amino acid scales only
		:return: dictionary with 5 values
		"""

        res = {}
        arc = PeptideDescriptor(self.ProteinSequence, scalename)
        arc.calculate_arc(modality)
        desc = arc.descriptor[0]
        for i in range(len(desc)):
            res['arc_' + str(i)] = desc[i]
        return res

    def calculate_autocorr(self, window, scalename='Eisenberg'):
        """
		Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp
		:param window: correlation window for descriptor calculation in a sliding window approach
		:param scalename:
		:return: dictionary with values of autocorrelation
		"""
        res = {}
        AMP = PeptideDescriptor(self.ProteinSequence, scalename)
        AMP.calculate_autocorr(window)
        desc = AMP.descriptor[0]
        for i in range(len(desc)):
            res['autocorr_' + str(i)] = desc[i]
        return res

    def calculate_crosscorr(self, window, scalename='Eisenberg'):
        """
		Calculates cross correlation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp
		:param window:correlation window for descriptor calculation in a sliding window approach
		:param scalename:
		:return: dictionary with values of crosscorrelation
		"""

        res = {}
        AMP = PeptideDescriptor(self.ProteinSequence, scalename)
        AMP.calculate_crosscorr(window)
        desc = AMP.descriptor[0]
        for i in range(len(desc)):
            res['crosscorr_' + str(i)] = desc[i]
        return res

    # ################# GET ALL FUNCTIONS ##################

    def get_all_physicochemical(self, ph=7, amide=False):
        """
		Calculate all 15 geral descriptors functions derived from biopython and modlpam
		:param ph: for functions Charge, charge density and formula
		:param amide: for functions Charge, charge density and formula
		:return: dictionary with variable number of descriptors
		"""
        res = {}
        res.update(self.get_lenght())
        res.update(self.get_charge(ph, amide))
        res.update(self.get_charge_density(ph, amide))
        res.update(self.get_formula(amide))
        res.update(self.get_bond())
        res.update(self.get_mw())
        # res.update(self.GetMW2())
        res.update(self.get_gravy())
        res.update(self.get_aromacity())
        res.update(self.get_isoelectric_point())
        # res.update(self.GetIsoelectricPoint2())
        res.update(self.get_instability_index())
        res.update(self.get_sec_struct())
        res.update(self.get_molar_extinction_coefficient())
        # res.update(self.get_flexibility())
        res.update(self.get_aliphatic_index())
        res.update(self.get_boman_index())
        res.update(self.get_hydrophobic_ratio())
        return res

    def get_all_aac(self):
        """
		Calculate all descriptors from Amino Acid Composition
		:return: dictionary with values from AAC, DPC and TPC
		"""
        res = {}
        res.update(self.get_aa_comp())
        res.update(self.get_dp_comp())
        res.update(self.get_tp_comp())
        return res

    def get_all_paac(self, lamda_paac=10, weight_paac=0.05, lamda_apaac=10, weight_apaac=0.05):
        """
		Calculate all descriptors from Pseudo Amino Acid Composition
		:param lamda_paac: parameter for PAAC default 10
		:param weight_paac: parameter for PAAC default 0.05
		:param lamda_apaac: parameter for APAAC default 10
		:param weight_apaac: parameter for APAAC default 0.05
		:return: dictionary with values from PAAC and APAAC
		"""
        res = {}
        lamda = lamda_paac
        weight = weight_paac
        res.update(self.get_paac(lamda, weight))
        lamda = lamda_apaac
        weight = weight_apaac
        res.update(self.get_apaac(lamda, weight))
        return res

    def get_all_sequenceorder(self, maxlag_socn=45, maxlag_qso=30, weight_qso=0.1):
        """
		Calculate all values for sequence order descriptors
		:param maxlag_socn: parameter for SOCN default 45
		:param maxlag_qso: parameter for QSO default 30
		:param weight_qso: parameter for QSO default 0.1
		:return: dictionary with values for quasi sequence order and sequence order couplig numbers
		"""

        res = {}

        maxlag = maxlag_socn
        res.update(self.get_socn(maxlag))

        maxlag = maxlag_qso
        weight = weight_qso
        res.update(self.get_qso(maxlag, weight))
        return res

    def get_all_correlation(self):
        """
		Calculate all descriptors from Autocorrelation
		:return: values for the funtions Moreau Broto, Moran and Geary autocorrelation
		"""
        res = {}
        res.update(self.get_moreau_broto_auto())
        res.update(self.get_moran_auto())
        res.update(self.get_geary_auto())
        return res

    def get_all_base_class(self, window=7, scalename='Eisenberg', scalename_arc='peparc', angle=100, modality='max',
                           prof_type='uH'):
        """
		Calculate all functions from Base class
		:param window:
		:param scalename:
		:param scalename_arc:
		:param angle:
		:param modality:
		:param prof_type:
		:return: dictionary with all 6 base class peptide descriptors (the value is variable)
		"""
        res = {}
        res.update(self.calculate_autocorr(window, scalename))
        res.update(self.calculate_crosscorr(window, scalename))
        res.update(self.calculate_moment(window, angle, modality, scalename))
        res.update(self.calculate_global(window, modality, scalename))
        res.update(self.calculate_profile(prof_type, window, scalename))
        res.update(self.calculate_arc(modality, scalename_arc))
        return res

    def get_all(self, ph=7, amide=False, tricomp=False, bin_aa=False, bin_prop=False, lamda_paac=10, weight_paac=0.05,
                lamda_apaac=10, weight_apaac=0.05, maxlag_socn=45, maxlag_qso=30, weight_qso=0.1, window=7,
                scalename='Eisenberg', scalename_arc='peparc', angle=100, modality='max', prof_type='uH'):

        """
		Calculate all descriptors from pydpi_py3 except tri-peptide pydpi_py3 and binary profiles
		:param ph:parameters for geral descriptors
		:param amide:parameters for geral descriptors
		:param tricomp: true or false to calculate or not tri-peptide pydpi_py3
		:param bin_aa: true or false to calculate or not binary composition of aa
		:param bin_prop: true or false to calculate or not binary composition of properties of resides
		:param lamda_paac: parameters for PAAC: lamdaPAAC=10 should not be larger than len(sequence)
		:param weight_paac: parameters for PAAC weightPAAC=0.05
		:param AAP: list with
		:param lamda_apaac: parmeters for APAAC lamdaAPAAC=10
		:param weight_apaac:parmeters for APAAC weightAPAAC=0.05
		:param maxlag_socn: parameters for SOCN: maxlagSOCN=45
		:param maxlag_qso:parameters for QSO maxlagQSO=30
		:param weight_qso:parameters for  weightQSO=0.1
		:param window:parameters for base class descriptors
		:param scalename:parameters for base class descriptors
		:param scalename_arc:parameters for base class descriptors
		:param angle:parameters for base class descriptors
		:param modality:parameters for base class descriptors
		:param prof_type:parameters for base class descriptors
		:return:dictionary with all features (value is variable)
		"""

        res = {}
        if bin_aa == True: res.update(self.get_bin_aa())
        if bin_prop == True: res.update(self.get_bin_resi_prop())
        # Geral
        res.update(self.get_lenght())
        res.update(self.get_charge(ph, amide))
        res.update(self.get_charge_density(ph, amide))
        res.update(self.get_formula(amide))
        res.update(self.get_bond())
        res.update(self.get_mw())
        # res.update(self.GetMW2())
        res.update(self.get_gravy())
        res.update(self.get_aromacity())
        res.update(self.get_isoelectric_point())
        # res.update(self.GetIsoelectricPoint2())
        res.update(self.get_instability_index())
        res.update(self.get_sec_struct())
        res.update(self.get_molar_extinction_coefficient())

        # res.update(self.get_flexibility()) (ver se é o q esta  causar nas)
        res.update(self.get_aliphatic_index())
        res.update(self.get_boman_index())
        res.update(self.get_hydrophobic_ratio())

        # pydpi_based
        res.update(self.get_aa_comp())
        res.update(self.get_dp_comp())
        if tricomp == True: res.update(self.get_tp_comp())
        res.update(self.get_moreau_broto_auto())
        res.update(self.get_moran_auto())
        res.update(self.get_geary_auto())

        res.update(self.get_ctd())
        res.update(self.get_conj_t())

        lamda = lamda_paac
        weight = weight_paac
        res.update(self.get_paac(lamda, weight))
        lamda = lamda_apaac
        weight = weight_apaac
        res.update(self.get_apaac(lamda, weight))

        maxlag = maxlag_socn
        res.update(self.get_socn(maxlag))

        maxlag = maxlag_qso
        weight = weight_qso
        res.update(self.get_qso(maxlag, weight))

        # base class
        res.update(self.calculate_autocorr(window, scalename))
        res.update(self.calculate_crosscorr(window, scalename))
        res.update(self.calculate_moment(window, angle, modality, scalename))
        res.update(self.calculate_global(window, modality, scalename))
        res.update(self.calculate_profile(prof_type, window, scalename))
        res.update(self.calculate_arc(modality, scalename_arc))
        return res

    def adaptable(self, list_of_functions=[], ph=7, amide=False, tricomp=False, lamda_paac=10, weight_paac=0.05,
                  lamda_apaac=10, weight_apaac=0.05, aap=[], maxlag_socn=45, maxlag_qso=30, weight_qso=0.1,
                  distancematrix={}, window=7, scalename='Eisenberg', scalename_arc='peparc', angle=100,
                  modality='max', prof_type='uH'):
        """
		Function to calculate user selected descriptors
		:param ph:parameters for geral descriptors
		:param amide:parameters for geral descriptors
		:param tricomp: true or false to calculate or not tri-peptide on get_all
		:param lamda_paac: parameters for PAAC: lamdaPAAC=10
		:param weight_paac: parameters for PAAC weightPAAC=0.05
		:param lamda_apaac: parmeters for APAAC lamdaAPAAC=5 IT SHOULD NOT BE LARGER THAN LENGHT SEQUENCE
		:param weight_apaac:parmeters for APAAC weightAPAAC=0.05
		:param aap:
		:param maxlag_socn: parameters for SOCN: maxlagSOCN=45
		:param maxlag_qso:parameters for QSO maxlagQSO=30
		:param weight_qso:parameters for  weightQSO=0.1
		:param distancematrix:
		:param window:parameters for base class descriptors
		:param scalename:parameters for base class descriptors
		:param scalename_arc:parameters for base class descriptors
		:param angle:parameters for base class descriptors
		:param modality:parameters for base class descriptors
		:param prof_type:parameters for base class descriptors
		:return:dictionary with all features (value is variable)
		:param list_of_functions:
		"""
        res = {}
        for function in list_of_functions:
            if function == 1: res.update(self.get_bin_aa())
            if function == 2: res.update(self.get_bin_resi_prop())

            if function == 3: res.update(self.get_lenght())
            if function == 4: res.update(self.get_charge(ph, amide))
            if function == 5: res.update(self.get_charge_density(ph, amide))
            if function == 6: res.update(self.get_formula(amide))
            if function == 7: res.update(self.get_bond())
            if function == 8: res.update(self.get_mw())
            if function == 9: res.update(self.get_gravy())
            if function == 10: res.update(self.get_aromacity())
            if function == 11: res.update(self.get_isoelectric_point())
            if function == 12: res.update(self.get_instability_index())
            if function == 13: res.update(self.get_sec_struct())
            if function == 14: res.update(self.get_molar_extinction_coefficient())
            if function == 16: res.update(self.get_flexibility())
            if function == 16: res.update(self.get_aliphatic_index())
            if function == 17: res.update(self.get_boman_index())
            if function == 18: res.update(self.get_hydrophobic_ratio())
            if function == 19: res.update(self.get_all_physicochemical(ph, amide))

            if function == 20: res.update(self.get_aa_comp())
            if function == 21: res.update(self.get_dp_comp())
            if function == 22: res.update(self.get_tp_comp())
            if function == 23: res.update(self.get_all_aac())

            lamda = lamda_paac
            weight = weight_paac
            if function == 24: res.update(self.get_paac(lamda, weight))
            if function == 25: res.update(self.get_paac_p(lamda, weight, aap))
            lamda = lamda_apaac
            weight = weight_apaac
            if function == 26: res.update(self.get_apaac(lamda, weight))
            if function == 27: res.update(self.get_all_paac(lamda_paac, weight_paac, lamda_apaac, weight_apaac))

            if function == 28: res.update(self.get_moreau_broto_auto())
            if function == 29: res.update(self.get_moran_auto())
            if function == 30: res.update(self.get_geary_auto())
            if function == 31: res.update(self.get_all_correlation())

            if function == 32: res.update(self.get_ctd())

            if function == 33: res.update(self.get_conj_t())

            maxlag = maxlag_socn
            if function == 34: res.update(self.get_socn(maxlag))
            if function == 35: res.update(self.get_socn_p(maxlag, distancematrix))
            maxlag = maxlag_qso
            weight = weight_qso
            if function == 36: res.update(self.get_qso(maxlag, weight))
            if function == 36: res.update(self.get_qso_p(maxlag, weight, distancematrix))
            if function == 37: res.update(self.get_all_sequenceorder(maxlag_socn, maxlag_qso, weight_qso))

            # base class
            if function == 38: res.update(self.calculate_autocorr(window, scalename))
            if function == 39: res.update(self.calculate_crosscorr(window, scalename))
            if function == 40: res.update(self.calculate_moment(window, angle, modality, scalename))
            if function == 41: res.update(self.calculate_global(window, modality, scalename))
            if function == 42: res.update(self.calculate_profile(prof_type, window, scalename))
            if function == 43: res.update(self.calculate_arc(modality, scalename_arc))
            if function == 44: res.update(self.get_all_base_class(window, scalename, scalename_arc))

            if function == 45: res.update(
                self.get_all(ph, amide, tricomp, lamda_paac, weight_paac, lamda_apaac, weight_apaac, maxlag_socn,
                             maxlag_qso,
                             weight_qso, window, scalename, scalename_arc, angle, modality, prof_type))

        return res
