"""
##############################################################################

A class  used for computing different types of protein descriptors parallelized.
It contains descriptors from packages pydpi, biopython, pfeature and modlamp.

Authors: Ana Marta Sequeira, Miguel Barros

Date: 05/2019 ALTERED 03/2022

Email:

##############################################################################
"""
import pandas as pd
from joblib import Parallel, delayed
from .adjuv_functions.features_functions.descriptors_modlamp import GlobalDescriptor, PeptideDescriptor
from .adjuv_functions.features_functions.bondcomp import boc_wp
from .adjuv_functions.features_functions.aa_composition import calculate_aa_composition, \
    calculate_dipeptide_composition, get_spectrum_dict
from .adjuv_functions.features_functions.pseudo_aac import get_pseudo_aac, get_a_pseudo_aac
from .adjuv_functions.features_functions.autocorrelation import \
    calculate_normalized_moreau_broto_auto_total, calculate_moran_auto_total, calculate_geary_auto_total
from .adjuv_functions.features_functions.ctd import calculate_ctd
from .adjuv_functions.features_functions.quasi_sequence_order import \
    get_sequence_order_coupling_number_total, get_quasi_sequence_order
from .adjuv_functions.features_functions.quasi_sequence_order import \
    get_sequence_order_coupling_numberp, get_quasi_sequence_orderp
from .adjuv_functions.features_functions.conjoint_triad import calculate_conjoint_triad
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ProteinDescritors:
    def __init__(self, dataset, col: str = 'sequence'):
        """
        Constructor

        :param dataset: the data corresponding to the protein sequences, it should be an string (one sequence),
         a list  or a pandas dataframe (multiples sequences).
        :param col: the name of the column in the dataframe which contains the protein sequences (pandas dataframe),
        or the name to give to the protein sequence column (list or string). Default collumn name is 'sequence'.
        """
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        elif isinstance(dataset, str):
            data = {col: [dataset]}
            self.dataset = pd.DataFrame(data)
        elif isinstance(dataset, list):
            data = {col: dataset}
            self.dataset = pd.DataFrame(data)
        else:
            raise Exception('Parameter dataframe is not an string, list or pandas Dataframe')

        self.col = col
        self.dataset.drop_duplicates(subset=self.col, keep='first', inplace=True)
        self.result = self.dataset

    def get_lenght(self, n_jobs: int = 4):
        """
        Calculates lenght of sequence (number of aa)

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of lenght for each sequence in the dataset
        """
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_lenght)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_charge(self, ph: float = 7.4, amide: bool = False, n_jobs: int = 4):
        """
        Calculates charge of sequence (1 value) from modlamp

        :param ph: ph considered to calculate. 7.4 by default
        :param amide: by default is not considered an amide protein sequence.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of charge for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_charge)(seq, self.col, ph, amide) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_charge_density(self, ph: float = 7.0, amide: bool = False, n_jobs: int = 4):
        """
        Calculates charge density of sequence (1 value) from modlamp

        :param ph: ph considered to calculate. 7 by default
        :param amide: by default is not considered an amide protein sequence.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of charge density for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_charge_density)(seq, self.col, ph, amide) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_formula(self, amide: bool = False, n_jobs: int = 4):
        """
        Calculates number of C,H,N,O and S of the aa of sequence (5 values) from modlamp

        :param amide: by default is not considered an amide protein sequence.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 5 values of C,H,N,O and S for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_formula)(seq, self.col, amide) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_bond(self, n_jobs: int = 4):
        """
        This function gives the sum of the bond composition for each type of bond
        For bond composition four types of bonds are considered
        total number of bonds (including aromatic), hydrogen bond, single bond and double bond.

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with 4 values for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_bond)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_mw(self, n_jobs: int = 4):
        """
        Calculates molecular weight of sequence (1 value) from modlamp

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of molecular weight for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_mw)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_gravy(self, n_jobs: int = 4):
        """
        Calculates Gravy from sequence (1 value) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of gravy for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_gravy)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_aromacity(self, n_jobs: int = 4):
        """
        Calculates Aromacity from sequence (1 value) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of aromacity for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_aromacity)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_isoelectric_point(self, n_jobs: int = 4):
        """
        Calculates Isolectric Point from sequence (1 value) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of Isolectric Point for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_isoelectric_point)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_instability_index(self, n_jobs: int = 4):
        """
        Calculates Instability index from sequence (1 value) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of Instability index  for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_instability_index)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_sec_struct(self, n_jobs: int = 4):
        """
        Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 3 value of helix, turn, sheet for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_sec_struct)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_molar_extinction_coefficient(self, n_jobs: int = 4):
        # [reduced, oxidized] # with reduced cysteines / # with disulfid bridges
        """
        Calculates the molar extinction coefficient (2 values) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of reduced cysteins and oxidized (with disulfid bridges) for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_molar_extinction_coefficient)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_flexibility(self, n_jobs: int = 4):
        """
        Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with proteinsequencelenght-9 values of flexiblity for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_flexibility)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_aliphatic_index(self, n_jobs: int = 4):
        """
        Calculates aliphatic index of sequence (1 value) from modlamp

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of aliphatic index for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_aliphatic_index)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_boman_index(self, n_jobs: int = 4):
        """
        Calculates boman index of sequence (1 value) from modlamp

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of boman index for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_boman_index)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_hydrophobic_ratio(self, n_jobs: int = 4):
        """
        Calculates hydrophobic ratio of sequence (1 value) from modlamp

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the value of hydrophobic ratio for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_hydrophobic_ratio)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    ################## AMINO ACID COMPOSITION ##################

    def get_aa_comp(self, n_jobs: int = 4):
        """
        Calculates amino acid compositon (20 values)  from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the fractions of all 20 aa(keys are the aa) for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_aa_comp)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_dp_comp(self, n_jobs: int = 4):
        """
        Calculates dipeptide composition (400 values) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the fractions of all 400 possible combiinations of 2 aa for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_dp_comp)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_tp_comp(self, n_jobs: int = 4):
        """
        Calculates tripeptide composition (8000 values) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the fractions of all 8000 possible combinations of 3 aa for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_tp_comp)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    ################## PSEUDO AMINO ACID COMPOSITION ##################

    def get_paac(self, lamda: int = 10, weight: float = 0.05, n_jobs: int = 4):
        """
        Calculates Type I Pseudo amino acid composition (default is 30, depends on lamda) from pydpi

        :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                    should NOT be larger than the length of input protein sequence
                    when lamda =0, the output of PseAA server is the 20-D amino acid composition
        :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                    The user can select any value within the region from 0.05 to 0.7 for the weight factor.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the fractions of all PAAC (keys are the PAAC) for each sequence in the dataset. Number of keys depends on lamda
        """
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_paac)(seq, self.col, lamda, weight) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_paac_p(self, lamda: int = 10, weight: float = 0.05, AAP=None, n_jobs: int = 4):
        """
        Calculates Type I Pseudo amino acid composition for a given property (default is 30, depends on lamda) from pydpi

        :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                    should NOT be larger than the length of input protein sequence
                    when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
        :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                    The user can select any value within the region from 0.05 to 0.7 for the weight factor.
        :param AAP: list of properties. each of which is a dict form.
                PseudoAAC._Hydrophobicity,PseudoAAC._hydrophilicity, PseudoAAC._residuemass,PseudoAAC._pK1,
                PseudoAAC._pK2,PseudoAAC._pI
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the fractions of all PAAC(keys are the PAAC) for each sequence in the dataset. Number of keys depends on lamda
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_paac_p)(seq, self.col, lamda, weight, AAP) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_apaac(self, lamda: int = 10, weight: float = 0.5, n_jobs: int = 4):
        """
        Calculates Type II Pseudo amino acid composition - Amphiphilic (default is 30, depends on lamda) from pydpi

        :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                    should NOT be larger than the length of input protein sequence
                    when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
        :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                    The user can select any value within the region from 0.05 to 0.7 for the weight factor.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the fractions of all PAAC(keys are the PAAC) for each sequence in the dataset. Number of keys depends on lamda
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_apaac)(seq, self.col, lamda, weight) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# AUTOCORRELATION DESCRIPTORS ##################

    def get_moreau_broto_auto(self, n_jobs: int = 4):
        """
        Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 240 descriptors for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_moreau_broto_auto)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_moran_auto(self, n_jobs: int = 4):
        """
        Calculates  Moran autocorrelation (240 values) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 240 descriptors for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_moran_auto)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_geary_auto(self, n_jobs: int = 4):
        """
        Calculates  Geary autocorrelation (240 values) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 240 descriptors for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_geary_auto)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

    def get_ctd(self, n_jobs: int = 4):
        """
        Calculates the Composition Transition Distribution descriptors (147 values) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 147 descriptors for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_ctd)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# CONJOINT TRIAD ##################

    def get_conj_t(self, n_jobs: int = 4):
        """
        Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the 343 descriptors for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_conj_t)(seq, self.col) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # #################  SEQUENCE ORDER  ##################

    def get_socn(self, maxlag: int = 45, n_jobs: int = 4):
        """
        Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi

        :param maxlag: maximum lag. Smaller than length of the protein
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the descriptors (90 descriptors) for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_socn)(seq, self.col, maxlag) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_socn_p(self, maxlag: int = 45, distancematrix=None, n_jobs: int = 4):
        """
        Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi

        :param maxlag: maximum lag. Smaller than length of the protein
        :param distancematrix: dict form containing 400 distance values
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the descriptors (90 descriptors) for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_socn_p)(seq, self.col, maxlag, distancematrix) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_qso(self, maxlag: int = 30, weight: float = 0.1, n_jobs: int = 4):
        """
        Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi

        :param maxlag: maximum lag. Smaller than length of the protein
        :param weight:
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the descriptors (100 descriptors) for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_qso)(seq, self.col, maxlag, weight) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def get_qso_p(self, maxlag: int = 30, weight: float = 0.1, distancematrix=None, n_jobs: int = 4,
                  ):
        """
        Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi

        :param maxlag: maximum lag. Smaller than length of the protein
        :param weight:
        :param distancematrix: dict form containing 400 distance values
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the descriptors (100 descriptors)
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(
                delayed(adjuv_qso_p)(seq, self.col, maxlag, weight, distancematrix) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

    """amino acid descriptor scales available are the ones from modlamo. 
    For more information please check: https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor
    amino acid sclaes include AASI, argos, bulkiness, charge_phys, charge_acid, eisenberg and others."""

    def calculate_moment(self, window: int = 1000, angle: int = 100, modality: str = 'max',
                         scalename: str = 'Eisenberg', n_jobs: int = 4):
        """
        Calculates moment of sequence (1 value) from modlamp

        :param window: amino acid window in which to calculate the moment. If the sequence is shorter than the window,
            the length of the sequence is taken
        :param angle: angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets
        :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with one value of moment for each sequence in the dataset
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")
        if angle != 100 and angle != 180: raise Exception(
            "Parameter angle must be 100 (alpha helices) or 180 (beta sheets)")
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_moment)(seq, self.col, window, angle, modality, scalename)
                           for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_global(self, window: int = 1000, modality: str = 'max', scalename: str = 'Eisenberg', n_jobs: int = 4,
                         ):
        """
        Calculates a global / window averaging descriptor value of a given AA scale of sequence (1 value) from modlamp

        :param window: size of sliding window used amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
        :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with one value for each sequence in the dataset
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_global)(seq, self.col, window, modality, scalename)
                           for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_profile(self, prof_type: str = 'uH', window: int = 7, scalename: str = 'Eisenberg', n_jobs: int = 4,
                          ):
        """
        Calculates hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercep
        (2 values) from modlamp

        :param prof_type: prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment
        :param window:size of sliding window used amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with two value for each sequence in the dataset
        """
        if prof_type != 'H' and prof_type != 'uH': raise Exception(
            "Parameter prof_type must be 'H' (hydrophobicity) or 'uH' (hydrophobic)")

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_profile)(seq, self.col, prof_type, window, scalename) for seq in
                           self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_arc(self, modality: str = "max", scalename: str = 'peparc', n_jobs: int = 4):
        """
        Calculates arcs as seen in the helical wheel plot. Use for binary amino acid scales only (5 values) from modlamp

        :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values only binary scales. By default peparc.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with 5 values for each sequence in the dataset
        """
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(
                delayed(adjuv_calculate_arc)(seq, self.col, modality, scalename)
                for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_autocorr(self, window: int = 7, scalename: str = 'Eisenberg', n_jobs: int = 4):
        """
        Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp

        :param window: correlation window for descriptor calculation in a sliding window approach
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with values of autocorrelation for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_autocorr)(seq, self.col, window, scalename) for seq in self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    def calculate_crosscorr(self, window: int = 7, scalename: str = 'Eisenberg', n_jobs: int = 4):
        """
        Calculates cross correlation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp

        :param window:correlation window for descriptor calculation in a sliding window approach
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with values of crosscorrelation for each sequence in the dataset
        """

        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(adjuv_calculate_crosscorr)(seq, self.col, window, scalename) for seq in
                           self.dataset[self.col])
        res = pd.DataFrame(res)
        return res

    # ################# GET ALL FUNCTIONS ##################

    def get_all_physicochemical(self, ph: float = 7, amide: bool = False, n_jobs: int = 4):
        """
        Calculate all 15 geral descriptors functions derived from biopython and modlpam

        :param ph: for functions Charge, charge density and formula
        :param amide: for functions Charge, charge density and formula
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with variable number of descriptors for each sequence in the dataset
        """
        self.result = self.result.merge(self.get_lenght(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_charge(ph, amide, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_charge_density(ph, amide, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_formula(amide, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_bond(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_mw(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_gravy(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_aromacity(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_isoelectric_point(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_instability_index(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_sec_struct(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_molar_extinction_coefficient(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_aliphatic_index(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_boman_index(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_hydrophobic_ratio(n_jobs=n_jobs), how='left', on = self.col)
        return self.result

    def get_all_aac(self, n_jobs: int = 4):
        """
        Calculate all descriptors from Amino Acid Composition

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with values from AAC, DPC and TP for each sequence in the dataset
        """
        self.result = self.result.merge(self.get_aa_comp(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_dp_comp(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_tp_comp(n_jobs=n_jobs), how='left', on = self.col)
        return self.result

    def get_all_paac(self, lamda_paac: int = 10, weight_paac: float = 0.05, lamda_apaac: int = 10,
                     weight_apaac: float = 0.05, n_jobs: int = 4):
        """
        Calculate all descriptors from Pseudo Amino Acid Composition

        :param lamda_paac: parameter for PAAC default 10
        :param weight_paac: parameter for PAAC default 0.05
        :param lamda_apaac: parameter for APAAC default 10
        :param weight_apaac: parameter for APAAC default 0.05
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with values from PAAC and APAAC  for each sequence in the dataset
        """
        self.result = self.result.merge(self.get_paac(lamda_paac, weight_paac, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.get_apaac(lamda_apaac, weight_apaac, n_jobs=n_jobs), how='left',
                                        on = self.col)
        return self.result

    def get_all_sequenceorder(self, maxlag_socn: int = 45, maxlag_qso: int = 30, weight_qso: float = 0.1,
                              n_jobs: int = 4):
        """
        Calculate all values for sequence order descriptors

        :param maxlag_socn: parameter for SOCN default 45
        :param maxlag_qso: parameter for QSO default 30
        :param weight_qso: parameter for QSO default 0.1
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with values for quasi sequence order and sequence order couplig numbers for each sequence
            in the dataset
        """
        self.result = self.result.merge(self.get_socn(maxlag_socn, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_qso(maxlag_qso, weight_qso, n_jobs=n_jobs), how='left', on = self.col)
        return self.result

    def get_all_correlation(self, n_jobs: int = 4):
        """
        Calculate all descriptors from Autocorrelation

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Datframe containing values for the funtions Moreau Broto, Moran and Geary autocorrelation for each
            sequence in the dataset
        """
        self.result = self.result.merge(self.get_moreau_broto_auto(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_moran_auto(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_geary_auto(n_jobs=n_jobs), how='left', on = self.col)
        return self.result

    def get_all_base_class(self, window: int = 7, scalename: str = 'Eisenberg', scalename_arc: str = 'peparc',
                           angle: int = 100, modality: str = 'max',
                           prof_type: str = 'uH', n_jobs: int = 4):
        """
        Calculate all functions from Base class

        :param window: size of sliding window used amino acid window. If the sequence is shorter than the window, the length of the sequence is taken.
        :param scalename: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
        :param scalename_arc: name of the amino acid scale (one in
        https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values only binary scales. By default peparc.
        :param angle: angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets
        :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
        :param prof_type: prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with all 6 base class peptide descriptors (the value is variable) for each sequence in the dataset
        """
        if prof_type != 'H' and prof_type != 'uH': raise Exception(
            "Parameter prof_type must be 'H' (hydrophobicity) or 'uH' (hydrophobic)")
        if modality != 'max' and modality != 'mean' and modality != 'all': raise Exception(
            "Parameter modality must be 'max' (maximum), 'mean' (mean) or 'all'")
        if angle != 100 and angle != 180: raise Exception(
            "Parameter angle must be 100 (alpha helices) or 180 (beta sheets)")

        self.result = self.result.merge(self.calculate_autocorr(window, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_crosscorr(window, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_moment(window, angle, modality, scalename, n_jobs=n_jobs),
                                        how='left', on = self.col)
        self.result = self.result.merge(self.calculate_global(window, modality, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_profile(prof_type, window, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_arc(modality, scalename_arc, n_jobs=n_jobs), how='left',
                                        on = self.col)
        return self.result

    def get_all(self, ph: float = 7, amide: bool = False, lamda_paac: int = 10,
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
        :param tricomp: true or false to calculate or not tri-peptide pydpi_py3
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores.
        :return: Dataframe with all features (value is variable)  for each sequence in the dataset
        """
        self.result = self.result.merge(self.get_lenght(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_charge(ph, amide, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_charge_density(ph, amide, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_formula(amide, n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_bond(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_mw(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_gravy(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_aromacity(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_isoelectric_point(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_instability_index(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_sec_struct(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_molar_extinction_coefficient(n_jobs=n_jobs), how='left', on = self.col)

        self.result = self.result.merge(self.get_aliphatic_index(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_boman_index(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_hydrophobic_ratio(n_jobs=n_jobs), how='left', on = self.col)

        # pydpi_base
        self.result = self.result.merge(self.get_aa_comp(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_dp_comp(n_jobs=n_jobs), how='left', on = self.col)
        if tricomp == True: self.result = self.result.merge(self.get_tp_comp(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_moreau_broto_auto(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_moran_auto(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_geary_auto(n_jobs=n_jobs), how='left', on = self.col)

        self.result = self.result.merge(self.get_ctd(n_jobs=n_jobs), how='left', on = self.col)
        self.result = self.result.merge(self.get_conj_t(n_jobs=n_jobs), how='left', on = self.col)

        self.result = self.result.merge(self.get_paac(lamda_paac, weight_paac, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.get_apaac(lamda_apaac, weight_apaac, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.get_socn(maxlag_socn, n_jobs=n_jobs), how='left', on = self.col)

        self.result = self.result.merge(self.get_qso(maxlag_qso, weight_qso, n_jobs=n_jobs), how='left', on = self.col)

        # base class
        self.result = self.result.merge(self.calculate_autocorr(window, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_crosscorr(window, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_moment(window, angle, modality, scalename, n_jobs=n_jobs),
                                        how='left', on = self.col)
        self.result = self.result.merge(self.calculate_global(window, modality, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_profile(prof_type, window, scalename, n_jobs=n_jobs), how='left',
                                        on = self.col)
        self.result = self.result.merge(self.calculate_arc(modality, scalename_arc, n_jobs=n_jobs), how='left',
                                        on = self.col)
        return self.result

    def get_adaptable(self, list_of_functions : list, ph: float = 7, amide: bool = False, lamda_paac: int = 10,
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
		:param AAP: list of properties. each of which is a dict form.
                PseudoAAC._Hydrophobicity,PseudoAAC._hydrophilicity, PseudoAAC._residuemass,PseudoAAC._pK1,
                PseudoAAC._pK2,PseudoAAC._pI
		:param maxlag_socn: parameters for SOCN: maxlagSOCN=45
		:param maxlag_qso:parameters for QSO maxlagQSO=30
		:param weight_qso:parameters for  weightQSO=0.1
		:param distancematrix: dict form containing 400 distance values
		:param window:parameters for base class descriptors
		:param scalename:parameters for base class descriptors
		:param scalename_arc:parameters for base class descriptors
		:param angle:parameters for base class descriptors
		:param modality:parameters for base class descriptors
		:param prof_type:parameters for base class descriptors
		:param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
		:return: Dataframe with the selected features for each sequence in the dataset
        """
        for function in list_of_functions:
            if function == 1: self.result = self.result.merge(self.get_lenght(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 2: self.result = self.result.merge(self.get_charge(ph, amide, n_jobs=n_jobs), how='left',
                                                              on = self.col)
            if function == 3: self.result = self.result.merge(self.get_charge_density(ph, amide, n_jobs=n_jobs),
                                                              how='left', on = self.col)
            if function == 4: self.result = self.result.merge(self.get_formula(amide, n_jobs=n_jobs), how='left',
                                                              on = self.col)
            if function == 5: self.result = self.result.merge(self.get_bond(n_jobs=n_jobs), how='left', on = self.col)
            if function == 6: self.result = self.result.merge(self.get_mw(n_jobs=n_jobs), how='left', on = self.col)
            if function == 7: self.result = self.result.merge(self.get_gravy(n_jobs=n_jobs), how='left', on = self.col)
            if function == 8: self.result = self.result.merge(self.get_aromacity(n_jobs=n_jobs), how='left',
                                                              on = self.col)
            if function == 9: self.result = self.result.merge(self.get_isoelectric_point(n_jobs=n_jobs), how='left',
                                                              on = self.col)
            if function == 10: self.result = self.result.merge(self.get_instability_index(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 11: self.result = self.result.merge(self.get_sec_struct(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 12: self.result = self.result.merge(self.get_molar_extinction_coefficient(n_jobs=n_jobs),
                                                               how='left', on = self.col)
            if function == 13: self.result = self.result.merge(self.get_flexibility(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 14: self.result = self.result.merge(self.get_aliphatic_index(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 15: self.result = self.result.merge(self.get_boman_index(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 16: self.result = self.result.merge(self.get_hydrophobic_ratio(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 17: self.get_all_physicochemical(ph, amide, n_jobs=n_jobs)

            if function == 18: self.result = self.result.merge(self.get_aa_comp(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 19: self.result = self.result.merge(self.get_dp_comp(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 20: self.result = self.result.merge(self.get_tp_comp(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 21: self.get_all_aac(n_jobs=n_jobs)

            if function == 22: self.result = self.result.merge(self.get_paac(lamda_paac, weight_paac, n_jobs=n_jobs),
                                                               how='left', on = self.col)
            if function == 23: self.result = self.result.merge(
                self.get_paac_p(lamda_paac, weight_paac, AAP, n_jobs=n_jobs), how='left', on = self.col)

            if function == 24: self.result = self.result.merge(self.get_apaac(lamda_apaac, weight_apaac, n_jobs=n_jobs),
                                                               how='left', on = self.col)
            if function == 25: self.get_all_paac(lamda_paac, weight_paac, lamda_apaac, weight_apaac, n_jobs=n_jobs)

            if function == 26: self.result = self.result.merge(self.get_moreau_broto_auto(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 27: self.result = self.result.merge(self.get_moran_auto(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 28: self.result = self.result.merge(self.get_geary_auto(n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 29: self.get_all_correlation(n_jobs=n_jobs)

            if function == 30: self.result = self.result.merge(self.get_ctd(n_jobs=n_jobs), how='left', on = self.col)

            if function == 31: self.result = self.result.merge(self.get_conj_t(n_jobs=n_jobs), how='left',
                                                               on = self.col)

            if function == 32: self.result = self.result.merge(self.get_socn(maxlag_socn, n_jobs=n_jobs), how='left',
                                                               on = self.col)
            if function == 33: self.result = self.result.merge(
                self.get_socn_p(maxlag_socn, distancematrix, n_jobs=n_jobs), how='left', on = self.col)

            if function == 34: self.result = self.result.merge(self.get_qso(maxlag_qso, weight_qso, n_jobs=n_jobs),
                                                               how='left', on = self.col)
            if function == 35: self.result = self.result.merge(
                self.get_qso_p(maxlag_qso, weight_qso, distancematrix, n_jobs=n_jobs), how='left', on = self.col)
            if function == 36: self.get_all_sequenceorder(maxlag_socn, maxlag_qso, weight_qso, n_jobs=n_jobs)

            # base class can take some time to run
            if function == 37: self.result = self.result.merge(
                self.calculate_autocorr(window, scalename, n_jobs=n_jobs), how='left', on = self.col)
            if function == 38: self.result = self.result.merge(
                self.calculate_crosscorr(window, scalename, n_jobs=n_jobs), how='left', on = self.col)
            if function == 39: self.result = self.result.merge(
                self.calculate_moment(window, angle, modality, scalename, n_jobs=n_jobs), how='left', on = self.col)
            if function == 40: self.result = self.result.merge(
                self.calculate_global(window, modality, scalename, n_jobs=n_jobs), how='left', on = self.col)
            if function == 41: self.result = self.result.merge(
                self.calculate_profile(prof_type, window, scalename, n_jobs=n_jobs), how='left', on = self.col)
            if function == 42: self.result = self.result.merge(
                self.calculate_arc(modality, scalename_arc, n_jobs=n_jobs), how='left', on = self.col)
            if function == 43: self.get_all_base_class(window, scalename, scalename_arc, angle, modality,
                                        prof_type, n_jobs=n_jobs)
            if function == 44:
                self.get_all(ph, amide, lamda_paac,weight_paac, lamda_apaac, weight_apaac, maxlag_socn,
                             maxlag_qso, weight_qso, window, scalename,scalename_arc, angle, modality,
                             prof_type, tricomp, n_jobs=n_jobs)

        return self.result


def adjuv_lenght(protein_sequence : str, col : str):
    """
    Calculates lenght of sequence (number of aa)

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of lenght
    """
    res = {col : protein_sequence}
    res['length'] = float(len(protein_sequence.strip()))
    return res


def adjuv_charge(protein_sequence : str, col : str, ph: float = 7.4, amide: bool = False):
    """
    Calculates charge of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param ph: ph considered to calculate. 7.4 by default
    :param amide: by default is not considered an amide protein sequence.
    :return: dictionary with the value of charge
    """
    res = {col : protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.calculate_charge(ph=ph, amide=amide)
    res['charge'] = desc.descriptor[0][0]
    return res


def adjuv_charge_density(protein_sequence : str, col : str, ph: float = 7.0, amide: bool = False):
    """
    Calculates charge density of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param ph: ph considered to calculate. 7 by default
    :param amide: by default is not considered an amide protein sequence.
    :return: dictionary with the value of charge density
    """

    res = {col : protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.charge_density(ph, amide)
    res['chargedensity'] = desc.descriptor[0][0]
    return res


def adjuv_formula(protein_sequence : str, col : str, amide: bool = False):
    """
    Calculates number of C,H,N,O and S of the aa of sequence (5 values) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param amide: by default is not considered an amide protein sequence.
    :return: dictionary with the 5 values of C,H,N,O and S
    """
    res = {col : protein_sequence}
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


def adjuv_bond(protein_sequence : str, col : str):
    """
    This function gives the sum of the bond composition for each type of bond
    For bond composition four types of bonds are considered
    total number of bonds (including aromatic), hydrogen bond, single bond and double bond.

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with 4 values
    """
    res = {col : protein_sequence}
    res.update(boc_wp(protein_sequence))
    return res


def adjuv_mw(protein_sequence : str, col : str):
    """
    Calculates molecular weight of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of molecular weight
    """

    res = {col : protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.calculate_MW(amide=True)
    res['MW_modlamp'] = desc.descriptor[0][0]
    return res


def adjuv_gravy(protein_sequence : str, col : str):
    """
    Calculates Gravy from sequence (1 value) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of gravy
    """

    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Gravy'] = analysed_seq.gravy()
    return res


def adjuv_aromacity(protein_sequence : str, col : str):
    """
    Calculates Aromacity from sequence (1 value) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of aromacity
    """

    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Aromacity'] = analysed_seq.aromaticity()
    return res


def adjuv_isoelectric_point(protein_sequence : str, col : str):
    """
    Calculates Isolectric Point from sequence (1 value) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of Isolectric Point
    """

    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['IsoelectricPoint'] = analysed_seq.isoelectric_point()
    return res


def adjuv_instability_index(protein_sequence : str, col : str):
    """
    Calculates Instability index from sequence (1 value) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of Instability index
    """
    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Instability_index'] = analysed_seq.instability_index()
    return res


def adjuv_sec_struct(protein_sequence : str, col : str):
    """
    Calculates the fraction of amino acids which tend to be in helix, turn or sheet (3 value) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the 3 value of helix, turn, sheet
    """
    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['SecStruct_helix'] = analysed_seq.secondary_structure_fraction()[0]  # helix
    res['SecStruct_turn'] = analysed_seq.secondary_structure_fraction()[1]  # turn
    res['SecStruct_sheet'] = analysed_seq.secondary_structure_fraction()[2]  # sheet
    return res


def adjuv_molar_extinction_coefficient(protein_sequence : str, col : str): # [reduced, oxidized] # with reduced cysteines / # with disulfid bridges
    """
    Calculates the molar extinction coefficient (2 values) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of reduced cysteins and oxidized (with disulfid bridges)
    """
    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    res['Molar_extinction_coefficient_reduced'] = analysed_seq.molar_extinction_coefficient()[0]  # reduced
    res['Molar_extinction_coefficient_oxidized'] = analysed_seq.molar_extinction_coefficient()[1]  # cys cys bounds
    return res


def adjuv_flexibility(protein_sequence : str, col : str):
    """
    Calculates the flexibility according to Vihinen, 1994 (return proteinsequencelenght-9 values ) from biopython

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with proteinsequencelenght-9 values of flexiblity
    """

    res = {col : protein_sequence}
    analysed_seq = ProteinAnalysis(protein_sequence)
    flexibility = analysed_seq.flexibility()
    for i in range(len(flexibility)):
        res['flexibility_' + str(i)] = flexibility[i]
    return res


def adjuv_aliphatic_index(protein_sequence : str, col : str):
    """
    Calculates aliphatic index of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of aliphatic index
    """
    res = {col : protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.aliphatic_index()
    res['aliphatic_index'] = desc.descriptor[0][0]
    return res


def adjuv_boman_index(protein_sequence : str, col : str):
    """
    Calculates boman index of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of boman index
    """
    res = {col : protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.boman_index()
    res['bomanindex'] = desc.descriptor[0][0]
    return res


def adjuv_hydrophobic_ratio(protein_sequence : str, col : str):
    """
    Calculates hydrophobic ratio of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the value of hydrophobic ratio
    """
    res = {col : protein_sequence}
    desc = GlobalDescriptor(protein_sequence)
    desc.hydrophobic_ratio()
    res['hydrophobic_ratio'] = desc.descriptor[0][0]
    return res


################## AMINO ACID COMPOSITION ##################

def adjuv_aa_comp(protein_sequence : str, col : str):
    """
    Calculates amino acid compositon (20 values)  from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the fractions of all 20 aa(keys are the aa)
    """
    res = {col : protein_sequence}
    res.update(calculate_aa_composition(protein_sequence))
    return res


def adjuv_dp_comp(protein_sequence : str, col : str):
    """
    Calculates dipeptide composition (400 values) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the fractions of all 400 possible combiinations of 2 aa
    """
    res = {col : protein_sequence}
    res.update(calculate_dipeptide_composition(protein_sequence))
    return res


def adjuv_tp_comp(protein_sequence : str, col : str):
    """
    Calculates tripeptide composition (8000 values) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the fractions of all 8000 possible combinations of 3 aa
    """
    res = {col : protein_sequence}
    res.update(get_spectrum_dict(protein_sequence))
    return res


################## PSEUDO AMINO ACID COMPOSITION ##################

def adjuv_paac(protein_sequence : str, col : str, lamda: int = 10, weight: float = 0.05):
    """
    Calculates Type I Pseudo amino acid composition (default is 30, depends on lamda) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                should NOT be larger than the length of input protein sequence
                when lamda =0, the output of PseAA server is the 20-D amino acid composition
    :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                The user can select any value within the region from 0.05 to 0.7 for the weight factor.
    :return: dictionary with the fractions of all PAAC (keys are the PAAC). Number of keys depends on lamda
    """
    res = {col : protein_sequence}
    res.update(get_pseudo_aac(protein_sequence, lamda=lamda, weight=weight))
    return res


def adjuv_paac_p(protein_sequence : str, col : str, lamda: int = 10, weight: float = 0.05, AAP=None):
    """
    Calculates Type I Pseudo amino acid composition for a given property (default is 30, depends on lamda) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                should NOT be larger than the length of input protein sequence
                when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
    :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                The user can select any value within the region from 0.05 to 0.7 for the weight factor.
    :param AAP: list of properties. each of which is a dict form.
            PseudoAAC._Hydrophobicity,PseudoAAC._hydrophilicity, PseudoAAC._residuemass,PseudoAAC._pK1,PseudoAAC._pK2,PseudoAAC._pI
    :return: dictionary with the fractions of all PAAC(keys are the PAAC). Number of keys depends on lamda
    """
    res = {col : protein_sequence}
    res.update(get_pseudo_aac(protein_sequence, lamda=lamda, weight=weight, AAP=AAP))
    return res


def adjuv_apaac(protein_sequence : str, col : str, lamda: int = 10, weight: float = 0.5):
    """
    Calculates Type II Pseudo amino acid composition - Amphiphilic (default is 30, depends on lamda) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param lamda: reflects the rank of correlation and is a non-Negative integer, such as 10.
                should NOT be larger than the length of input protein sequence
                when lamda =0, theoutput of PseAA server is the 20-D amino acid composition
    :param weight: weight on the additional PseAA components. with respect to the conventional AA components.
                The user can select any value within the region from 0.05 to 0.7 for the weight factor.
    :return: dictionary with the fractions of all PAAC(keys are the PAAC). Number of keys depends on lamda
    """
    res = {col : protein_sequence}
    res.update(get_a_pseudo_aac(protein_sequence, lamda=lamda, weight=weight))
    return res


# ################# AUTOCORRELATION DESCRIPTORS ##################

def adjuv_moreau_broto_auto(protein_sequence : str, col : str):
    """
    Calculates Normalized Moreau-Broto autocorrelation (240 values) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the 240 descriptors
    """
    res = {col : protein_sequence}
    res.update(calculate_normalized_moreau_broto_auto_total(protein_sequence))
    return res


def adjuv_moran_auto(protein_sequence : str, col : str):
    """
    Calculates  Moran autocorrelation (240 values) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the 240 descriptors
    """
    res = {col : protein_sequence}
    res.update(calculate_moran_auto_total(protein_sequence))
    return res


def adjuv_geary_auto(protein_sequence : str, col : str):
    """
    Calculates  Geary autocorrelation (240 values) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the 240 descriptors
    """
    res = {col : protein_sequence}
    res.update(calculate_geary_auto_total(protein_sequence))
    return res


# ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################

def adjuv_ctd(protein_sequence : str, col : str):
    """
    Calculates the Composition Transition Distribution descriptors (147 values) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the 147 descriptors
    """
    res = {col : protein_sequence}
    res.update(calculate_ctd(protein_sequence))
    return res


# ################# CONJOINT TRIAD ##################

def adjuv_conj_t(protein_sequence : str, col : str):
    """
    Calculates the Conjoint Triad descriptors (343 descriptors) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :return: dictionary with the 343 descriptors
    """
    res = {col : protein_sequence}
    res.update(calculate_conjoint_triad(protein_sequence))
    return res


# #################  SEQUENCE ORDER  ##################

def adjuv_socn(protein_sequence : str, col : str, maxlag: int = 45):
    """
    Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param maxlag: maximum lag. Smaller than length of the protein
    :return: dictionary with the descriptors (90 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {col : protein_sequence}
    res.update(get_sequence_order_coupling_number_total(protein_sequence, maxlag=maxlag))
    return res


def adjuv_socn_p(protein_sequence : str, col : str, maxlag: int = 45, distancematrix=None):
    """
    Calculates the Sequence order coupling numbers  (retrieves 90 values by default) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param maxlag: maximum lag. Smaller than length of the protein
    :param distancematrix: dict form containing 400 distance values
    :return: dictionary with the descriptors (90 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {col : protein_sequence}
    res.update(get_sequence_order_coupling_numberp(protein_sequence, maxlag=maxlag, distancematrix=distancematrix))
    return res


def adjuv_qso(protein_sequence : str, col : str, maxlag: int = 30, weight: float = 0.1):
    """
    Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param maxlag: maximum lag. Smaller than length of the protein
    :param weight:
    :return: dictionary with the descriptors (100 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {col : protein_sequence}
    res.update(get_quasi_sequence_order(protein_sequence, maxlag=maxlag, weight=weight))
    return res


def adjuv_qso_p(protein_sequence : str, col : str, maxlag: int = 30, weight: float = 0.1, distancematrix=None):
    """
    Calculates the Quasi sequence order  (retrieves 100 values by default) from pydpi

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param maxlag: maximum lag. Smaller than length of the protein
    :param weight:
    :param distancematrix: dict form containing 400 distance values
    :return: dictionary with the descriptors (100 descriptors)
    """
    if maxlag > len(protein_sequence): raise Exception(
        "Parameter maxlag must be smaller than length of the protein")
    res = {col : protein_sequence}
    res.update(get_quasi_sequence_orderp(protein_sequence, maxlag=maxlag, weight=weight,
                                         distancematrix=distancematrix))
    return res


# ################# BASE CLASS PEPTIDE DESCRIPTOR ##################

def adjuv_calculate_moment(protein_sequence : str, col : str, window: int = 1000, angle: int = 100, modality: str = 'max',
                           scalename: str = 'Eisenberg'):
    """
    Calculates moment of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param window: amino acid window in which to calculate the moment. If the sequence is shorter than the window,
            the length of the sequence is taken
    :param angle: angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets
    :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
    :param scalename: name of the amino acid scale (one in
    https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
    :return: dictionary with one value of moment
    """
    res = {col : protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_moment(window, angle, modality)
    res['moment'] = AMP.descriptor[0][0]
    return res


def adjuv_calculate_global(protein_sequence : str, col : str, window: int = 1000, modality: str = 'max', scalename: str = 'Eisenberg'):
    """
    Calculates a global / window averaging descriptor value of a given AA scale of sequence (1 value) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param window: size of sliding window used amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
    :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
    :param scalename: name of the amino acid scale (one in
    https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
    :return: dictionary with one value
    """
    res = {col : protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_global(window, modality)
    res['global'] = AMP.descriptor[0][0]
    return res


def adjuv_calculate_profile(protein_sequence : str, col : str, prof_type: str = 'uH', window: int = 7, scalename: str = 'Eisenberg'):
    """
    Calculates hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercep
    (2 values) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param prof_type: prof_type of profile, ‘H’ for hydrophobicity or ‘uH’ for hydrophobic moment
    :param window:size of sliding window used amino acid window. If the sequence is shorter than the window, the length of the sequence is taken
    :param scalename: name of the amino acid scale (one in
    https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
    :return: dictionary with two value
    """
    res = {col : protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_profile(prof_type, window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['profile_' + str(i)] = desc[i]
    return res


def adjuv_calculate_arc(protein_sequence : str, col : str, modality: str = "max", scalename: str = 'peparc'):
    """
    Calculates arcs as seen in the helical wheel plot. Use for binary amino acid scales only (5 values) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param modality: maximum (max), mean (mean) or both (all) hydrophobic moment
    :param scalename: name of the amino acid scale (one in
    https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values only binary scales. By default peparc.
    :return: dictionary with 5 values
    """
    res = {col : protein_sequence}
    arc = PeptideDescriptor(protein_sequence, scalename)
    arc.calculate_arc(modality)
    desc = arc.descriptor[0]
    for i in range(len(desc)):
        res['arc_' + str(i)] = desc[i]
    return res


def adjuv_calculate_autocorr(protein_sequence : str, col : str, window: int = 7, scalename: str = 'Eisenberg'):
    """
    Calculates autocorrelation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param window: correlation window for descriptor calculation in a sliding window approach
    :param scalename: name of the amino acid scale (one in
    https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
    :return: dictionary with values of autocorrelation
    """
    res = {col : protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_autocorr(window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['autocorr_' + str(i)] = desc[i]
    return res


def adjuv_calculate_crosscorr(protein_sequence : str, col : str, window: int = 7, scalename: str = 'Eisenberg'):
    """
    Calculates cross correlation of amino acid values for a given descriptor scale ( variable >>>>>>values) from modlamp

    :protein_sequence: Protein sequence to be processed
    :param col: name of the collum for the original protein sequences.
    :param window:correlation window for descriptor calculation in a sliding window approach
    :param scalename: name of the amino acid scale (one in
    https://modlamp.org/modlamp.html#modlamp.descriptors.PeptideDescriptor) used to calculate the descriptor values. By default Eisenberg.
    :return: dictionary with values of crosscorrelation
    """
    res = {col : protein_sequence}
    AMP = PeptideDescriptor(protein_sequence, scalename)
    AMP.calculate_crosscorr(window)
    desc = AMP.descriptor[0]
    for i in range(len(desc)):
        res['crosscorr_' + str(i)] = desc[i]
    return res
