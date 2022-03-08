"""
##############################################################################

A class  used for computing different types of protein descriptors parallelized.
It contains descriptors from packages pydpi, biopython, pfeature and modlamp.

Authors: Miguel Barros

Date: 03/2022

Email:

##############################################################################
"""
import pandas as pd
from joblib import Parallel, delayed
from propythia.adjuv_functions.features_functions.descriptors_modlamp import GlobalDescriptor, PeptideDescriptor
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

class ParDescritors:

    def __init__(self, dataset, col: str):
        """
        Constructor

        :param dataset: Pandas dataframe
        :param col: column in the dataframe which contains the protein sequence
        """
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else:
            raise Exception('Parameter dataframe must be a pandas dataframe')
        self.col = col
        self.result = dataset

    def par_lenght(self, n_jobs: int = 4):
        """
        Calculates lenght of sequence (number of aa)
        :return: dictionary with the value of lenght
        """
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_lenght)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_charge(self, ph: float = 7.4, amide: bool = False, n_jobs: int = 4):
        """
        Calculates charge of sequence (1 value) from modlamp
        :param ph: ph considered to calculate. 7.4 by default
        :param amide: by default is not considered an amide protein sequence.
        :return: dictionary with the value of charge
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_charge)(seq, ph, amide) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_charge_density(self, ph: float = 7.0, amide: bool = False, n_jobs: int = 4):
        """
        Calculates charge density of sequence (1 value) from modlamp
        :param ph: ph considered to calculate. 7 by default
        :param amide: by default is not considered an amide protein sequence.
        :return: dictionary with the value of charge density
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_charge_density)(seq, ph, amide) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_formula(self, amide: bool = False, n_jobs: int = 4):
        """
        Calculates number of C,H,N,O and S of the aa of sequence (5 values) from modlamp
        :param amide: by default is not considered an amide protein sequence.
        :return: dictionary with the 5 values of C,H,N,O and S
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_formula)(seq, amide) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_bond(self, n_jobs: int = 4):
        """
        This function gives the sum of the bond composition for each type of bond
        For bond composition four types of bonds are considered
        total number of bonds (including aromatic), hydrogen bond, single bond and double bond.
        :return: dictionary with 4 values
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_bond)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_mw(self, n_jobs: int = 4):
        """
        Calculates molecular weight of sequence (1 value) from modlamp
        :return: dictionary with the value of molecular weight
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_mw)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_gravy(self, n_jobs: int = 4):
        """
        Calculates Gravy from sequence (1 value) from biopython
        :return: dictionary with the value of gravy
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_gravy)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_aromacity(self, n_jobs: int = 4):
        """
        Calculates Aromacity from sequence (1 value) from biopython
        :return: dictionary with the value of aromacity
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_aromacity)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_isoelectric_point(self, n_jobs: int = 4):
        """
        Calculates Isolectric Point from sequence (1 value) from biopython
        :return: dictionary with the value of Isolectric Point
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_isoelectric_point)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_instability_index(self, n_jobs: int = 4):
        """
        Calculates Instability index from sequence (1 value) from biopython
        :return: dictionary with the value of Instability index
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_instability_index)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_sec_struct(self, n_jobs: int = 4):
        """
        Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython
        :return: dictionary with the 3 value of helix, turn, sheet
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_sec_struct)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_molar_extinction_coefficient(self, n_jobs: int = 4):
        # [reduced, oxidized] # with reduced cysteines / # with disulfid bridges
        """
        Calculates the molar extinction coefficient (2 values) from biopython
        :return: dictionary with the value of reduced cysteins and oxidized (with disulfid bridges)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_molar_extinction_coefficient)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_flexibility(self, n_jobs: int = 4):
        """
        Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython
        :return: dictionary with proteinsequencelenght-9 values of flexiblity
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_flexibility)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_aliphatic_index(self, n_jobs: int = 4):
        """
        Calculates aliphatic index of sequence (1 value) from modlamp
        :return: dictionary with the value of aliphatic index
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_aliphatic_index)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_boman_index(self, n_jobs: int = 4):
        """
        Calculates boman index of sequence (1 value) from modlamp
        :return: dictionary with the value of boman index
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_boman_index)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_hydrophobic_ratio(self, n_jobs: int = 4):
        """
        Calculates hydrophobic ratio of sequence (1 value) from modlamp
        :return: dictionary with the value of hydrophobic ratio
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_hydrophobic_ratio)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    ################## AMINO ACID COMPOSITION ##################

    def par_aa_comp(self, n_jobs: int = 4):
        """
        Calculates amino acid compositon (20 values)  from pydpi
        :return: dictionary with the fractions of all 20 aa(keys are the aa)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_aa_comp)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_dp_comp(self, n_jobs: int = 4):
        """
        Calculates dipeptide composition (400 values) from pydpi
        :return: dictionary with the fractions of all 400 possible combiinations of 2 aa
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_dp_comp)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_tp_comp(self, n_jobs: int = 4):
        """
            Calculates tripeptide composition (8000 values) from pydpi
            :return: dictionary with the fractions of all 8000 possible combinations of 3 aa
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_tp_comp)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    ################## PSEUDO AMINO ACID COMPOSITION ##################

    def par_paac(self, lamda: int = 10, weight: float = 0.05, n_jobs: int = 4):
        """
        Calculates Type I Pseudo amino acid composition (default is 30, depends on lamda) from pydpi
        :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                    should NOT be larger than the length of input protein sequence
                    when lamda =0, the output of PseAA server is the 20-D amino acid composition
        :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                    The user can select any value within the region from 0.05 to 0.7 for the weight factor.
        :return: dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
        """
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_paac)(seq, lamda, weight) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_paac_p(self, lamda: int = 10, weight: float = 0.05, AAP=None, n_jobs: int = 4):
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

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_paac_p)(seq, lamda, weight, AAP) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_apaac(self, lamda: int = 10, weight: float = 0.5, n_jobs: int = 4):
        """
        Calculates Type II Pseudo amino acid composition - Amphiphilic (default is 30, depends on lamda) from pydpi
        :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                    should NOT be larger than the length of input protein sequence
                    when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
        :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                    The user can select any value within the region from 0.05 to 0.7 for the weight factor.
        :return: dictionary with the fractions of all PAAC(keys are the PAAC). Number of keys depends on lamda
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_apaac)(seq, lamda, weight) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# AUTOCORRELATION DESCRIPTORS ##################

    def par_moreau_broto_auto(self, n_jobs: int = 4):
        """
        Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi
        :return: dictionary with the 240 descriptors
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_moreau_broto_auto)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_moran_auto(self, n_jobs: int = 4):
        """
        Calculates  Moran autocorrelation (240 values) from pydpi
        :return: dictionary with the 240 descriptors
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_moran_auto)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_geary_auto(self, n_jobs: int = 4):
        """
        Calculates  Geary autocorrelation (240 values) from pydpi
        :return: dictionary with the 240 descriptors
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_geary_auto)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

    def par_ctd(self, n_jobs: int = 4):
        """
        Calculates the Composition Transition Distribution descriptors (147 values) from pydpi
        :return: dictionary with the 147 descriptors
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_ctd)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# CONJOINT TRIAD ##################

    def par_conj_t(self, n_jobs: int = 4):
        """
        Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi
        :return: dictionary with the 343 descriptors
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_conj_t)(seq) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # #################  SEQUENCE ORDER  ##################

    def par_socn(self, maxlag: int = 45, n_jobs: int = 4):
        """
        Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi
        :param maxlag: maximum lag. Smaller than length of the protein
        :return: dictionary with the descriptors (90 descriptors)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_socn)(seq, maxlag) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_socn_p(self, maxlag: int = 45, distancematrix=None, n_jobs: int = 4):
        """
        Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi
        :param maxlag: maximum lag. Smaller than length of the protein
        :param distancematrix: dict form containing 400 distance values
        :return: dictionary with the descriptors (90 descriptors)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_socn_p)(seq, maxlag, distancematrix) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_qso(self, maxlag: int = 30, weight: float = 0.1, n_jobs: int = 4):
        """
        Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi
        :param maxlag: maximum lag. Smaller than length of the protein
        :param weight:
        :return: dictionary with the descriptors (100 descriptors)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_qso)(seq, maxlag, weight) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def par_qso_p(self, maxlag: int = 30, weight: float = 0.1, distancematrix=None, n_jobs: int = 4,
            ):
        """
        Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi
        :param maxlag: maximum lag. Smaller than length of the protein
        :param weight:
        :param distancematrix: dict form containing 400 distance values
        :return: dictionary with the descriptors (100 descriptors)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(
                delayed(adjuv_qso_p)(seq, maxlag, weight, distancematrix) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

    """amino acid descriptor scales available are the ones from modlamo. 
    For more information please check:  https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor
    amino acid sclaes include AASI, argos, bulkiness, charge_phys, charge_acid, eisenberg and others."""

    def calculate_moment(self, window: int = 1000, angle: int = 100, modality: str = 'max',
                         scalename: str = 'Eisenberg', n_jobs: int = 4):
        """
        Calculates moment of sequence (1 value) from modlamp
        :param window: amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken
        :param angle: angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets
        :param modality: maximum or mean hydrophobic moment
        :param scalename:
        :return: dictionary with one value of moment
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")
        if angle != 100 and angle != 180: raise Exception(
            "Parameter angle must be 100 (alpha helices) or 180 (beta sheets)")
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")


        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_moment)(seq, window, angle, modality, scalename)
                           for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_global(self, window: int = 1000, modality: str = 'max', scalename: str = 'Eisenberg', n_jobs: int = 4,
                    ):
        """
        Calculates a global / window averaging descriptor value of a given AA scale of sequence (1 value) from modlamp
        :param window: amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
        :param modality: maximum or mean hydrophobic moment
        :param scalename:
        :return: dictionary with one value
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")


        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_global)(seq, window, modality, scalename)
                           for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_profile(self, prof_type: str = 'uH', window: int = 7, scalename: str = 'Eisenberg', n_jobs: int = 4,
                    ):
        """
        Calculates hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercep
        (2 values) from modlamp
        :param prof_type: prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment
        :param window: size of sliding window used (odd-numbered)
        :param scalename:
        :return: dictionary with two value
        """
        if prof_type != 'H' and prof_type != 'uH': raise Exception(
            "Parameter prof_type must be 'H' (hydrophobicity) or 'uH' (hydrophobic)")


        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_profile)(seq, prof_type, window, scalename) for seq in
                           self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_arc(self, modality: str = "max", scalename: str = 'peparc', n_jobs: int = 4):
        """
        Calculates arcs as seen in the helical wheel plot. Use for binary amino acid scales only (5 values) from modlamp
        :param modality: maximum or mean
        :param scalename: binary amino acid scales only
        :return: dictionary with 5 values
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(
                delayed(adjuv_calculate_arc)(seq, modality, scalename)
                for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_autocorr(self, window: int = 7, scalename: str = 'Eisenberg', n_jobs: int = 4):
        """
        Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp
        :param window: correlation window for descriptor calculation in a sliding window approach
        :param scalename:
        :return: dictionary with values of autocorrelation
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_autocorr)(seq, window, scalename) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_crosscorr(self, window: int = 7, scalename: str = 'Eisenberg', n_jobs: int = 4):
        """
        Calculates cross correlation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp
        :param window:correlation window for descriptor calculation in a sliding window approach
        :param scalename:
        :return: dictionary with values of crosscorrelation
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_crosscorr)(seq, window, scalename) for seq in
                           self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# GET ALL FUNCTIONS ##################

    def par_all_physicochemical(self, ph: float = 7, amide: bool = False, n_jobs: int = 4):
        """
        Calculate all 15 geral descriptors functions derived from biopython and modlpam
        :param ph: for functions Charge, charge density and formula
        :param amide: for functions Charge, charge density and formula
        :return: dictionary with variable number of descriptors
        """
        self.par_lenght(n_jobs=n_jobs)
        self.par_charge(ph, amide, n_jobs=n_jobs)
        self.par_charge_density(ph, amide, n_jobs=n_jobs)
        self.par_formula(amide, n_jobs=n_jobs)
        self.par_bond(n_jobs=n_jobs)
        self.par_mw(n_jobs=n_jobs)
        self.par_gravy(n_jobs=n_jobs)
        self.par_aromacity(n_jobs=n_jobs)
        self.par_isoelectric_point(n_jobs=n_jobs)
        self.par_instability_index(n_jobs=n_jobs)
        self.par_sec_struct(n_jobs=n_jobs)
        self.par_molar_extinction_coefficient(n_jobs=n_jobs)
        self.par_aliphatic_index(n_jobs=n_jobs)
        self.par_boman_index(n_jobs=n_jobs)
        self.par_hydrophobic_ratio(n_jobs=n_jobs)
        return self.result

    def par_all_aac(self, n_jobs: int = 4):
        """
        Calculate all descriptors from Amino Acid Composition
        :return: dictionary with values from AAC, DPC and TPC
        """
        self.par_aa_comp(n_jobs=n_jobs)
        self.par_dp_comp(n_jobs=n_jobs)
        self.par_tp_comp(n_jobs=n_jobs)
        return self.result

    def par_all_paac(self, lamda_paac: int = 10, weight_paac: float = 0.05, lamda_apaac: int = 10,
                     weight_apaac: float = 0.05, n_jobs: int = 4):
        """
        Calculate all descriptors from Pseudo Amino Acid Composition
        :param lamda_paac: parameter for PAAC default 10
        :param weight_paac: parameter for PAAC default 0.05
        :param lamda_apaac: parameter for APAAC default 10
        :param weight_apaac: parameter for APAAC default 0.05
        :return: dictionary with values from PAAC and APAAC
        """
        self.par_paac(lamda_paac, weight_paac, n_jobs=n_jobs)
        self.par_apaac(lamda_apaac, weight_apaac, n_jobs=n_jobs)
        return self.result

    def par_all_sequenceorder(self, maxlag_socn: int = 45, maxlag_qso: int = 30, weight_qso: float = 0.1,
                              n_jobs: int = 4):
        """
        Calculate all values for sequence order descriptors
        :param maxlag_socn: parameter for SOCN default 45
        :param maxlag_qso: parameter for QSO default 30
        :param weight_qso: parameter for QSO default 0.1
        :return: dictionary with values for quasi sequence order and sequence order couplig numbers
        """
        self.par_socn(maxlag_socn, n_jobs=n_jobs)
        self.par_qso(maxlag_qso, weight_qso, n_jobs=n_jobs)
        return self.result

    def par_all_correlation(self, n_jobs: int = 4):
        """
        Calculate all descriptors from Autocorrelation
        :return: values for the funtions Moreau Broto, Moran and Geary autocorrelation
        """
        self.par_moreau_broto_auto(n_jobs=n_jobs)
        self.par_moran_auto(n_jobs=n_jobs)
        self.par_geary_auto(n_jobs=n_jobs)
        return self.result

    def par_all_base_class(self, window: int = 7, scalename: str = 'Eisenberg', scalename_arc: str = 'peparc',
                           angle: int = 100, modality: str = 'max',
                           prof_type: str = 'uH', n_jobs: int = 4):
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
        if prof_type != 'H' and prof_type != 'uH': raise Exception(
            "Parameter prof_type must be 'H' (hydrophobicity) or 'uH' (hydrophobic)")
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")
        if angle != 100 and angle != 180: raise Exception(
            "Parameter angle must be 100 (alpha helices) or 180 (beta sheets)")

        self.calculate_autocorr(window, scalename, n_jobs=n_jobs)
        self.calculate_crosscorr(window, scalename, n_jobs=n_jobs)
        self.calculate_moment(window, angle, modality, scalename, n_jobs=n_jobs)
        self.calculate_global(window, modality, scalename, n_jobs=n_jobs)
        self.calculate_profile(prof_type, window, scalename, n_jobs=n_jobs)
        self.calculate_arc(modality, scalename_arc, n_jobs=n_jobs)
        return self.result

    def par_all(self, ph: float = 7, amide: bool = False, lamda_paac: int = 10,
                weight_paac: float = 0.05, lamda_apaac: int = 10, weight_apaac: float = 0.05, maxlag_socn: int = 45,
                maxlag_qso: int = 30, weight_qso: float = 0.1, window: int = 7,
                scalename: str = 'Eisenberg', scalename_arc: str = 'peparc', angle: int = 100,
                modality: str = 'max',
                prof_type: str = 'uH', tricomp: bool = False, n_jobs: int = 4):

        """
        Calculate all descriptors from pydpi_py3 except tri-peptide pydpi_py3 and binary profiles
        :param ph:parameters for geral descriptors
        :param amide:parameters for geral descriptors
        :param lamda_paac: parameters for PAAC: lamdaPAAC=10 should not be larger than len(sequence)
        :param weight_paac: parameters for PAAC weightPAAC=0.05
        :param lamda_apaac: parameters for APAAC lamdaAPAAC=10
        :param weight_apaac:parameters for APAAC weightAPAAC=0.05
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
        :param tricomp: true or false to calculate or not tri-peptide pydpi_py3
        """
        self.par_lenght(n_jobs=n_jobs)
        self.par_charge(ph, amide, n_jobs=n_jobs)
        self.par_charge_density(ph, amide, n_jobs=n_jobs)
        self.par_formula(amide, n_jobs=n_jobs)
        self.par_bond(n_jobs=n_jobs)
        self.par_mw(n_jobs=n_jobs)
        self.par_gravy(n_jobs=n_jobs)
        self.par_aromacity(n_jobs=n_jobs)
        self.par_isoelectric_point(n_jobs=n_jobs)
        self.par_instability_index(n_jobs=n_jobs)
        self.par_sec_struct(n_jobs=n_jobs)
        self.par_molar_extinction_coefficient(n_jobs=n_jobs)

        self.par_aliphatic_index(n_jobs=n_jobs)
        self.par_boman_index(n_jobs=n_jobs)
        self.par_hydrophobic_ratio(n_jobs=n_jobs)

        # pydpi_base
        self.par_aa_comp(n_jobs=n_jobs)
        self.par_dp_comp(n_jobs=n_jobs)
        if tricomp == True: self.par_tp_comp(n_jobs=n_jobs)
        self.par_moreau_broto_auto(n_jobs=n_jobs)
        self.par_moran_auto(n_jobs=n_jobs)
        self.par_geary_auto(n_jobs=n_jobs)

        self.par_ctd(n_jobs=n_jobs)
        self.par_conj_t(n_jobs=n_jobs)

        self.par_paac(lamda_paac, weight_paac, n_jobs=n_jobs)
        self.par_apaac(lamda_apaac, weight_apaac, n_jobs=n_jobs)
        self.par_socn(maxlag_socn, n_jobs=n_jobs)

        self.par_qso(maxlag_qso, weight_qso, n_jobs=n_jobs)

        # base class
        self.calculate_autocorr(window, scalename, n_jobs=n_jobs)
        self.calculate_crosscorr(window, scalename, n_jobs=n_jobs)
        self.calculate_moment(window, angle, modality, scalename, n_jobs=n_jobs)
        self.calculate_global(window, modality, scalename, n_jobs=n_jobs)
        self.calculate_profile(prof_type, window, scalename, n_jobs=n_jobs)
        self.calculate_arc(modality, scalename_arc, n_jobs=n_jobs)
        return self.result

    def par_adaptable(self, list_of_functions=None, ph: float = 7, amide: bool = False, lamda_paac: int = 10,
                      weight_paac: float = 0.05, lamda_apaac: int = 10, weight_apaac: float = 0.05, AAP=None,
                      maxlag_socn: int = 45, maxlag_qso: int = 30, weight_qso: float = 0.1, distancematrix=None,
                      window: int = 7, scalename: str = 'Eisenberg', scalename_arc: str = 'peparc',
                      angle: int = 100, modality: str = 'max', prof_type: str = 'uH', tricomp: bool = False,
                      n_jobs: int = 4):
        """
        Function for parallelize the calculation of the user selected descriptors

		:param list_of_functions: list of functions desired to calculate descriptors. Numeration in the descriptors guide.
		:param ph:parameters for geral descriptors
		:param amide:parameters for geral descriptors
		:param lamda_paac: parameters for PAAC: lamdaPAAC=10
		:param weight_paac: parameters for PAAC weightPAAC=0.05
		:param lamda_apaac: parmeters for APAAC lamdaAPAAC=5 IT SHOULD NOT BE LARGER THAN LENGHT SEQUENCE
		:param weight_apaac:parmeters for APAAC weightAPAAC=0.05
		:param AAP:
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
		:param n_jobs: number of CPU cores to be used.
		:return: pandas dataframe with all features
        """
        for function in list_of_functions:
            if function == 1:  self.result = self.result.merge(self.par_lenght(n_jobs=n_jobs), how='left', on='sequence')
            if function == 2: self.result = self.result.merge(self.par_charge(ph, amide, n_jobs=n_jobs), how='left', on='sequence')
            if function == 3: self.result = self.result.merge(self.par_charge_density(ph, amide, n_jobs=n_jobs), how='left', on='sequence')
            if function == 4: self.result = self.result.merge(self.par_formula(amide, n_jobs=n_jobs), how='left', on='sequence')
            if function == 5: self.result = self.result.merge(self.par_bond(n_jobs=n_jobs), how='left', on='sequence')
            if function == 6: self.result = self.result.merge(self.par_mw(n_jobs=n_jobs), how='left', on='sequence')
            if function == 7: self.result = self.result.merge(self.par_gravy(n_jobs=n_jobs), how='left', on='sequence')
            if function == 8: self.result = self.result.merge(self.par_aromacity(n_jobs=n_jobs), how='left', on='sequence')
            if function == 9: self.result = self.result.merge(self.par_isoelectric_point(n_jobs=n_jobs), how='left', on='sequence')
            if function == 10: self.result = self.result.merge(self.par_instability_index(n_jobs=n_jobs), how='left', on='sequence')
            if function == 11: self.result = self.result.merge(self.par_sec_struct(n_jobs=n_jobs), how='left', on='sequence')
            if function == 12: self.result = self.result.merge(self.par_molar_extinction_coefficient(n_jobs=n_jobs), how='left', on='sequence')
            if function == 13: self.result = self.result.merge(self.par_flexibility(n_jobs=n_jobs), how='left', on='sequence')
            if function == 14: self.result = self.result.merge(self.par_aliphatic_index(n_jobs=n_jobs), how='left', on='sequence')
            if function == 15: self.result = self.result.merge(self.par_boman_index(n_jobs=n_jobs), how='left', on='sequence')
            if function == 16: self.result = self.result.merge(self.par_hydrophobic_ratio(n_jobs=n_jobs), how='left', on='sequence')
            if function == 17: self.result = self.result.merge(self.par_all_physicochemical(ph, amide, n_jobs=n_jobs), how='left', on='sequence')

            if function == 18: self.result = self.result.merge(self.par_aa_comp(n_jobs=n_jobs), how='left', on='sequence')
            if function == 19: self.result = self.result.merge(self.par_dp_comp(n_jobs=n_jobs), how='left', on='sequence')
            if function == 20: self.result = self.result.merge(self.par_tp_comp(n_jobs=n_jobs), how='left', on='sequence')
            if function == 21: self.result = self.result.merge(self.par_all_aac(n_jobs=n_jobs), how='left', on='sequence')

            if function == 22: self.result = self.result.merge(self.par_paac(lamda_paac, weight_paac, n_jobs=n_jobs), how='left', on='sequence')
            if function == 23: self.result = self.result.merge(self.par_paac_p(lamda_paac, weight_paac, AAP, n_jobs=n_jobs), how='left', on='sequence')

            if function == 24: self.result = self.result.merge(self.par_apaac(lamda_apaac, weight_apaac, n_jobs=n_jobs), how='left', on='sequence')
            if function == 25: self.result = self.result.merge(self.par_all_paac(lamda_paac, weight_paac, lamda_apaac, weight_apaac, n_jobs=n_jobs), how='left', on='sequence')

            if function == 26: self.result = self.result.merge(self.par_moreau_broto_auto(n_jobs=n_jobs), how='left', on='sequence')
            if function == 27: self.result = self.result.merge(self.par_moran_auto(n_jobs=n_jobs), how='left', on='sequence')
            if function == 28: self.result = self.result.merge(self.par_geary_auto(n_jobs=n_jobs), how='left', on='sequence')
            if function == 29: self.result = self.result.merge(self.par_all_correlation(n_jobs=n_jobs), how='left', on='sequence')

            if function == 30: self.result = self.result.merge(self.par_ctd(n_jobs=n_jobs), how='left', on='sequence')

            if function == 31: self.result = self.result.merge(self.par_conj_t(n_jobs=n_jobs), how='left', on='sequence')

            if function == 32: self.result = self.result.merge(self.par_socn(maxlag_socn, n_jobs=n_jobs), how='left', on='sequence')
            if function == 33: self.result = self.result.merge(self.par_socn_p(maxlag_socn, distancematrix, n_jobs=n_jobs), how='left', on='sequence')

            if function == 34: self.result = self.result.merge(self.par_qso(maxlag_qso, weight_qso, n_jobs=n_jobs), how='left', on='sequence')
            if function == 35: self.result = self.result.merge(self.par_qso_p(maxlag_qso, weight_qso, distancematrix, n_jobs=n_jobs), how='left', on='sequence')
            if function == 36: self.result = self.result.merge(self.par_all_sequenceorder(maxlag_socn, maxlag_qso, weight_qso, n_jobs=n_jobs), how='left', on='sequence')

            # base class can take some time to run
            if function == 37: self.result = self.result.merge(self.calculate_autocorr(window, scalename, n_jobs=n_jobs), how='left', on='sequence')
            if function == 38: self.result = self.result.merge(self.calculate_crosscorr(window, scalename, n_jobs=n_jobs), how='left', on='sequence')
            if function == 39: self.result = self.result.merge(self.calculate_moment(window, angle, modality, scalename, n_jobs=n_jobs), how='left', on='sequence')
            if function == 40: self.result = self.result.merge(self.calculate_global(window, modality, scalename, n_jobs=n_jobs), how='left', on='sequence')
            if function == 41: self.result = self.result.merge(self.calculate_profile(prof_type, window, scalename, n_jobs=n_jobs), how='left', on='sequence')
            if function == 42: self.result = self.result.merge(self.calculate_arc(modality, scalename_arc, n_jobs=n_jobs), how='left', on='sequence')
            if function == 43: self.result = self.result.merge(self.par_all_base_class(window, scalename, scalename_arc, angle, modality,
                                                       prof_type, n_jobs=n_jobs), how='left', on='sequence')
            if function == 44:
                self.result = self.result.merge(self.par_all(ph, amide, lamda_paac,
                             weight_paac, lamda_apaac, weight_apaac, maxlag_socn,
                             maxlag_qso, weight_qso, window, scalename,
                             scalename_arc, angle, modality,
                             prof_type, tricomp, n_jobs=n_jobs), how='left', on='sequence')

        return self.result



def adjuv_lenght(protein_sequence):
    """
    Calculates lenght of sequence (number of aa)
    :return: dictionary with the value of lenght
    """
    res = {'sequence': protein_sequence}
    res['length'] = float(len(protein_sequence.strip()))
    return res

def adjuv_charge(protein_sequence, ph: float = 7.4, amide: bool = False):
    """
    Calculates charge of sequence (1 value) from modlamp
    :param ph: ph considered to calculate. 7.4 by default
    :param amide: by default is not considered an amide protein sequence.
    :return: dictionary with the value of charge
    """
    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.calculate_charge(ph=ph, amide=amide)
    res['charge'] = desc.descriptor[0][0]
    return res

def adjuv_charge_density(protein_sequence, ph: float = 7.0, amide: bool = False):
    """
    Calculates charge density of sequence (1 value) from modlamp
    :param ph: ph considered to calculate. 7 by default
    :param amide: by default is not considered an amide protein sequence.
    :return: dictionary with the value of charge density
    """

    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.charge_density(ph, amide)
    res['chargedensity'] = desc.descriptor[0][0]
    return res

def adjuv_formula(protein_sequence, amide: bool = False):
    """
    Calculates number of C,H,N,O and S of the aa of sequence (5 values) from modlamp
    :param amide: by default is not considered an amide protein sequence.
    :return: dictionary with the 5 values of C,H,N,O and S
    """
    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
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

def adjuv_bond(protein_sequence):
    """
    This function gives the sum of the bond composition for each type of bond
    For bond composition four types of bonds are considered
    total number of bonds (including aromatic), hydrogen bond, single bond and double bond.
    :return: dictionary with 4 values
    """
    res = {'sequence': protein_sequence}
    res.update(boc_wp(protein_sequence))
    return res

def adjuv_mw(protein_sequence):
    """
    Calculates molecular weight of sequence (1 value) from modlamp
    :return: dictionary with the value of molecular weight
    """

    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.calculate_MW(amide=True)
    res['MW_modlamp'] = desc.descriptor[0][0]
    return res

def adjuv_gravy(protein_sequence):
    """
    Calculates Gravy from sequence (1 value) from biopython
    :return: dictionary with the value of gravy
    """

    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Gravy'] = analysed_seq.gravy()
    return res

def adjuv_aromacity(protein_sequence):
    """
    Calculates Aromacity from sequence (1 value) from biopython
    :return: dictionary with the value of aromacity
    """

    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Aromacity'] = analysed_seq.aromaticity()
    return res

def adjuv_isoelectric_point(protein_sequence):
    """
    Calculates Isolectric Point from sequence (1 value) from biopython
    :return: dictionary with the value of Isolectric Point
    """

    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['IsoelectricPoint'] = analysed_seq.isoelectric_point()
    return res

def adjuv_instability_index(protein_sequence):
    """
    Calculates Instability index from sequence (1 value) from biopython
    :return: dictionary with the value of Instability index
    """
    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Instability_index'] = analysed_seq.instability_index()
    return res

def adjuv_sec_struct(protein_sequence):
    """
    Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython
    :return: dictionary with the 3 value of helix, turn, sheet
    """
    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['SecStruct_helix'] = analysed_seq.secondary_structure_fraction()[0]  # helix
    res['SecStruct_turn'] = analysed_seq.secondary_structure_fraction()[1]  # turn
    res['SecStruct_sheet'] = analysed_seq.secondary_structure_fraction()[2]  # sheet
    return res

def adjuv_molar_extinction_coefficient(protein_sequence):  # [reduced, oxidized] # with reduced cysteines / # with disulfid bridges
    """
    Calculates the molar extinction coefficient (2 values) from biopython
    :return: dictionary with the value of reduced cysteins and oxidized (with disulfid bridges)
    """
    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Molar_extinction_coefficient_reduced'] = analysed_seq.molar_extinction_coefficient()[0]  # reduced
    res['Molar_extinction_coefficient_oxidized'] = analysed_seq.molar_extinction_coefficient()[1]  # cys cys bounds
    return res

def adjuv_flexibility(protein_sequence):
    """
    Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython
    :return: dictionary with proteinsequencelenght-9 values of flexiblity
    """

    res = {'sequence': protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    flexibility = analysed_seq.flexibility()
    for i in range(len(flexibility)):
        res['flexibility_' + str(i)] = flexibility[i]
    return res

def adjuv_aliphatic_index(protein_sequence):
    """
    Calculates aliphatic index of sequence (1 value) from modlamp
    :return: dictionary with the value of aliphatic index
    """
    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.aliphatic_index()
    res['aliphatic_index'] = desc.descriptor[0][0]
    return res

def adjuv_boman_index(protein_sequence):
    """
    Calculates boman index of sequence (1 value) from modlamp
    :return: dictionary with the value of boman index
    """
    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.boman_index()
    res['bomanindex'] = desc.descriptor[0][0]
    return res

def adjuv_hydrophobic_ratio(protein_sequence):
    """
    Calculates hydrophobic ratio of sequence (1 value) from modlamp
    :return: dictionary with the value of hydrophobic ratio
    """
    res = {'sequence': protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.hydrophobic_ratio()
    res['hydrophobic_ratio'] = desc.descriptor[0][0]
    return res

################## AMINO ACID COMPOSITION ##################

def adjuv_aa_comp(protein_sequence):
    """
    Calculates amino acid compositon (20 values)  from pydpi
    :return: dictionary with the fractions of all 20 aa(keys are the aa)
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_aa_composition(protein_sequence))
    return res

def adjuv_dp_comp(protein_sequence):
    """
    Calculates dipeptide composition (400 values) from pydpi
    :return: dictionary with the fractions of all 400 possible combiinations of 2 aa
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_dipeptide_composition(protein_sequence))
    return res

def adjuv_tp_comp(protein_sequence):
    """
        Calculates tripeptide composition (8000 values) from pydpi
        :return: dictionary with the fractions of all 8000 possible combinations of 3 aa
    """
    res = {'sequence': protein_sequence}
    res.update(get_spectrum_dict(protein_sequence))
    return res

################## PSEUDO AMINO ACID COMPOSITION ##################

def adjuv_paac(protein_sequence, lamda: int = 10, weight: float = 0.05):
    """
    Calculates Type I Pseudo amino acid composition (default is 30, depends on lamda) from pydpi
    :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                should NOT be larger than the length of input protein sequence
                when lamda =0, the output of PseAA server is the 20-D amino acid composition
    :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                The user can select any value within the region from 0.05 to 0.7 for the weight factor.
    :return: dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
    """
    res = {'sequence': protein_sequence}
    res.update(get_pseudo_aac(protein_sequence, lamda=lamda, weight=weight))
    return res

def adjuv_paac_p(protein_sequence, lamda: int = 10, weight: float = 0.05, AAP=None):
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
    res = {'sequence': protein_sequence}
    res.update(get_pseudo_aac(protein_sequence, lamda=lamda, weight=weight, AAP=AAP))
    return res

def adjuv_apaac(protein_sequence, lamda: int = 10, weight: float = 0.5):
    """
    Calculates Type II Pseudo amino acid composition - Amphiphilic (default is 30, depends on lamda) from pydpi
    :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                should NOT be larger than the length of input protein sequence
                when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
    :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                The user can select any value within the region from 0.05 to 0.7 for the weight factor.
    :return: dictionary with the fractions of all PAAC(keys are the PAAC). Number of keys depends on lamda
    """
    res = {'sequence': protein_sequence}
    res.update(get_a_pseudo_aac(protein_sequence, lamda=lamda, weight=weight))
    return res

# ################# AUTOCORRELATION DESCRIPTORS ##################

def adjuv_moreau_broto_auto(protein_sequence):
    """
    Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi
    :return: dictionary with the 240 descriptors
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_normalized_moreau_broto_auto_total(protein_sequence))
    return res

def adjuv_moran_auto(protein_sequence):
    """
    Calculates  Moran autocorrelation (240 values) from pydpi
    :return: dictionary with the 240 descriptors
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_moran_auto_total(protein_sequence))
    return res

def adjuv_geary_auto(protein_sequence):
    """
    Calculates  Geary autocorrelation (240 values) from pydpi
    :return: dictionary with the 240 descriptors
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_geary_auto_total(protein_sequence))
    return res

# ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

def adjuv_ctd(protein_sequence):
    """
    Calculates the Composition Transition Distribution descriptors (147 values) from pydpi
    :return: dictionary with the 147 descriptors
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_ctd(protein_sequence))
    return res

# ################# CONJOINT TRIAD ##################

def adjuv_conj_t( protein_sequence):
    """
    Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi
    :return: dictionary with the 343 descriptors
    """
    res = {'sequence': protein_sequence}
    res.update(calculate_conjoint_triad(protein_sequence))
    return res

# #################  SEQUENCE ORDER  ##################

def adjuv_socn(protein_sequence, maxlag: int = 45):
    """
    Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi
    :param maxlag: maximum lag. Smaller than length of the protein
    :return: dictionary with the descriptors (90 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {'sequence': protein_sequence}
    res.update(get_sequence_order_coupling_number_total(protein_sequence, maxlag=maxlag))
    return res

def adjuv_socn_p(protein_sequence, maxlag: int = 45, distancematrix=None):
    """
    Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi
    :param maxlag: maximum lag. Smaller than length of the protein
    :param distancematrix: dict form containing 400 distance values
    :return: dictionary with the descriptors (90 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {'sequence': protein_sequence}
    res.update(get_sequence_order_coupling_numberp(protein_sequence, maxlag=maxlag, distancematrix=distancematrix))
    return res

def adjuv_qso(protein_sequence, maxlag: int = 30, weight: float = 0.1):
    """
    Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi
    :param maxlag: maximum lag. Smaller than length of the protein
    :param weight:
    :return: dictionary with the descriptors (100 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {'sequence': protein_sequence}
    res.update(get_quasi_sequence_order(protein_sequence, maxlag=maxlag, weight=weight))
    return res

def adjuv_qso_p(protein_sequence, maxlag: int = 30, weight: float = 0.1, distancematrix=None):
    """
    Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi
    :param maxlag: maximum lag. Smaller than length of the protein
    :param weight:
    :param distancematrix: dict form containing 400 distance values
    :return: dictionary with the descriptors (100 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {'sequence': protein_sequence}
    res.update(get_quasi_sequence_orderp(protein_sequence, maxlag=maxlag, weight=weight,
                                    distancematrix=distancematrix))
    return res

# ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

def adjuv_calculate_moment(protein_sequence, window: int = 1000, angle: int = 100, modality: str = 'max',
                     scalename: str = 'Eisenberg'):
    """
    Calculates moment of sequence (1 value) from modlamp
    :param window: amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken
    :param angle: angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets
    :param modality: maximum or mean hydrophobic moment
    :param scalename:
    :return: dictionary with one value of moment
    """
    res = {'sequence': protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_moment(window, angle, modality)
    res['moment'] = AMP.descriptor[0][0]
    return res

def adjuv_calculate_global(protein_sequence, window: int = 1000, modality: str = 'max', scalename: str = 'Eisenberg'):
    """
    Calculates a global / window averaging descriptor value of a given AA scale of sequence (1 value) from modlamp
    :param window: amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
    :param modality: maximum or mean hydrophobic moment
    :param scalename:
    :return: dictionary with one value
    """
    res = {'sequence': protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_global(window, modality)
    res['global'] = AMP.descriptor[0][0]
    return res

def adjuv_calculate_profile(protein_sequence, prof_type: str = 'uH', window: int = 7, scalename: str = 'Eisenberg'):
    """
    Calculates hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercep
    (2 values) from modlamp
    :param prof_type: prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment
    :param window: size of sliding window used (odd-numbered)
    :param scalename:
    :return: dictionary with two value
    """
    res = {'sequence': protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_profile(prof_type, window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['profile_' + str(i)] = desc[i]
    return res

def adjuv_calculate_arc(protein_sequence, modality: str = "max", scalename: str = 'peparc'):
    """
    Calculates arcs as seen in the helical wheel plot. Use for binary amino acid scales only (5 values) from modlamp
    :param modality: maximum or mean
    :param scalename: binary amino acid scales only
    :return: dictionary with 5 values
    """
    res = {'sequence': protein_sequence}
    arc = PeptideDescriptor(protein_sequence, scalename)
    arc.calculate_arc(modality)
    desc = arc.descriptor[0]
    for i in range(len(desc)):
        res['arc_' + str(i)] = desc[i]
    return res

def adjuv_calculate_autocorr(protein_sequence, window: int = 7, scalename: str = 'Eisenberg'):
    """
    Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp
    :param window: correlation window for descriptor calculation in a sliding window approach
    :param scalename:
    :return: dictionary with values of autocorrelation
    """
    res = {'sequence': protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_autocorr(window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['autocorr_' + str(i)] = desc[i]
    return res

def adjuv_calculate_crosscorr(protein_sequence, window: int = 7, scalename: str = 'Eisenberg'):
    """
    Calculates cross correlation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp
    :param window:correlation window for descriptor calculation in a sliding window approach
    :param scalename:
    :return: dictionary with values of crosscorrelation
    """
    res = {'sequence': protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_crosscorr(window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['crosscorr_' + str(i)] = desc[i]
    return res
