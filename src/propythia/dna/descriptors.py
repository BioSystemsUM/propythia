"""
##########################################################################################

A class used for computing different types of DNA descriptors for a single DNA sequence.
It contains descriptors from packages iLearn, iDNA4mC, rDNAse, ...

Authors: João Nuno Abreu
Date: 02/2022
Email:

##########################################################################################
"""

from .utils import *
from functools import reduce
from typing import Dict, List, Tuple

class DNADescriptor:
    """
    The Descriptor class collects all descriptor calculation functions into a simple class.
    It returns the features in a dictionary object
    """

    def __init__(self, dna_sequence):
        # it is assumed that the sequence is a string with valid alphabet
        self.dna_sequence = dna_sequence
            

    def get_length(self) -> int:
        """
        Returns the length of the sequence. From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Parameters
        ----------
        
        Returns
        -------
        int
            The length of the sequence.
        """
        return len(self.dna_sequence)

    def get_gc_content(self) -> float:
        """
        Returns the GC content of the sequence.
        Parameters
        ----------
        
        Returns
        -------
        float
            The GC content of the sequence.
        """
        gc_content = 0
        for letter in self.dna_sequence:
            if letter == 'G' or letter == 'C':
                gc_content += 1
        return round(gc_content / self.get_length(), 3)

    def get_at_content(self) -> float:
        """
        Returns the AT content of the sequence.
        Parameters
        ----------

        Returns
        -------
        float
            The AT content of the sequence.
        """
        at_content = 0
        for letter in self.dna_sequence:
            if letter == 'A' or letter == 'T':
                at_content += 1
        return round(at_content / self.get_length(), 3)

    # ----------------------- NUCLEIC ACID COMPOSITION ----------------------- #

    def get_nucleic_acid_composition(self, normalize: bool = True) -> Dict[str, float]:
        """
        Calculates the Nucleic Acid Composition of the sequence. From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Parameters
        ----------
        normalize : bool (default=True)
            Default value is False. If True, this method returns the frequencies of each nucleic acid.
        Returns
        -------
        Dict of str:float
            Dictionary with values of nucleic acid composition
        """
        res = make_kmer_dict(1)
        for letter in self.dna_sequence:
            res[letter] += 1

        if normalize:
            res = normalize_dict(res)
        return res

    def get_dinucleotide_composition(self, normalize: bool = True) -> Dict[str, float]:
        """
        Calculates the Dinucleotide Composition of the sequence. From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Parameters
        ----------
        normalize : bool (default=True)
            Default value is False. If True, this method returns the frequencies of each dinucleotide.
        Returns
        -------
        Dict of str:float
            Dictionary with values of dinucleotide composition
        """
        res = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i+2]
            res[dinucleotide] += 1
        if normalize:
            res = normalize_dict(res)
        return res

    def get_trinucleotide_composition(self, normalize: bool = True) -> Dict[str, float]:
        """
        Calculates the Trinucleotide Composition of the sequence. From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Parameters
        ----------
        normalize : bool (default=True)
            Default value is False. If True, this method returns the frequencies of each trinucleotide.
        Returns
        -------
        Dict of str:float
            Dictionary with values of trinucleotide composition
        """
        res = make_kmer_dict(3)
        for i in range(len(self.dna_sequence) - 2):
            trinucleotide = self.dna_sequence[i:i+3]
            res[trinucleotide] += 1

        if normalize:
            res = normalize_dict(res)
        return res

    def get_k_spaced_nucleic_acid_pairs(self, k: int = 0, normalize: bool = True) -> Dict[str, float]:
        """
        Calculates the K-Spaced Nucleic Acid Pairs of the sequence. From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Parameters
        ----------
        k : int (default=0)
            The number of nucleic acids to pair together.
        normalize: bool (default=True)
            Default value is False. If True, this method returns the frequencies of each k-spaced nucleic acid pair.
        Returns
        -------
        Dict of str:float
            Dictionary with values of k-spaced nucleic acid pairs
        """
        res = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - k - 1):
            k_spaced_nucleic_acid_pair = self.dna_sequence[i] + \
                self.dna_sequence[i+k+1]
            res[k_spaced_nucleic_acid_pair] += 1

        if normalize:
            res = normalize_dict(res)
        return res

    def get_kmer(self, k: int = 2, normalize: bool = True, reverse: bool = False) -> Dict[str, float]:
        """
        Calculates the K-Mer of the sequence. From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://rdrr.io/cran/rDNAse/
        Parameters 
        ----------
        k : int (default=2)
            The number of nucleic acids to pair together.
        normalize: bool (default=True)
            Default value is False. If True, this method returns the frequencies of all kmers.
        reverse : bool (default=False)
            Whether to calculate the reverse complement kmer or not.
        Returns
        -------
        Dict of str:float
            Dictionary with values of kmer
        """
        
        res = make_kmer_dict(k)

        for i in range(len(self.dna_sequence) - k + 1):
            res[self.dna_sequence[i:i+k]] += 1

        if reverse:
            for kmer, _ in sorted(res.items(), key=lambda x: x[0]):
                reverse_val = "".join([pairs[i] for i in kmer[::-1]])

                # calculate alphabet order between kmer and reverse compliment
                if(kmer < reverse_val):
                    smaller = kmer
                    bigger = reverse_val
                else:
                    smaller = reverse_val
                    bigger = kmer

                # create in dict if they dont exist
                if(smaller not in res):
                    res[smaller] = 0
                if(bigger not in res):
                    res[bigger] = 0

                if(smaller != bigger):
                    # add to dict
                    res[smaller] += res[bigger]
                    # delete from dict
                    del res[bigger]

        if normalize:
            res = normalize_dict(res)

        return res
    
    def get_accumulated_nucleotide_frequency(self, normalize: bool = True) -> List[Dict[str, float]]:
        """
        Calculates the Accumulated Nucleotide Frequency of the sequence at 25%, 50% and 75%. From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://www.nature.com/articles/srep13859?proof=t%252Btarget%253D
        Parameters
        ----------
        normalize: bool (default=True)
            Default value is False. If True, this method returns the frequencies of all accumulated nucleotide frequencies.
        Returns
        -------
        list of dicts of str:float
            The Accumulated Nucleotide Frequency of the sequence at 25%, 50% and 75%.
        """
        res = []
        d1 = make_kmer_dict(1)
        d2 = make_kmer_dict(1)
        d3 = make_kmer_dict(1)
        
        for letter in self.dna_sequence[:normal_round(len(self.dna_sequence) * 0.25)]:
            d1[letter] += 1
            
        for letter in self.dna_sequence[:normal_round(len(self.dna_sequence) * 0.50)]:
            d2[letter] += 1
        
        for letter in self.dna_sequence[:normal_round(len(self.dna_sequence) * 0.75)]:
            d3[letter] += 1
        res = [d1,d2,d3]

        if normalize:
            res = [normalize_dict(d1),normalize_dict(d2),normalize_dict(d3)]
        return res

    # --------------------------  Autocorrelation  -------------------------- #

    def get_DAC(self, phyche_index: List[str] = ["Twist", "Tilt"], nlag: int = 2, all_property: bool = False, extra_phyche_index: Dict[str, List[Tuple[str, float]]] = None) -> List[float]:
        """
        Calculates the Dinucleotide Based Auto Covariance of the sequence. CODE FROM repDNA (https://github.com/liufule12/repDNA)
        Parameters
        ----------
        phyche_index : list of str (default=["Twist", "Tilt"])
            The physicochemical properties list.
        nlag : int (default=2)
            An integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA
            sequence in the dataset). It represents the distance between two dinucleotides.
        all_property : bool (default=False)
            If True, returns all properties.
        extra_phyche_index : dict of str and list of float (default=None)
            The extra phyche index to use for the calculation. It means user-defined phyche_index. The key is
            dinucleotide, and its corresponding value is a list with a pair of physicochemical indices and its new
            value.
        Returns
        -------
        list of float
            The Dinucleotide Based Auto Covariance of the sequence.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_ac_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_DCC(self, phyche_index: List[str] = ["Twist", "Tilt"], nlag: int = 2, all_property: bool = False, extra_phyche_index: Dict[str, List[Tuple[str, float]]] = None) -> List[float]:
        """
        Calculates the Dinucleotide Based Cross Covariance (DCC) of the sequence. CODE FROM repDNA (https://github.com/liufule12/repDNA)
        Parameters
        ----------
        phyche_index : list of str (default=["Twist", "Tilt"])
            The physicochemical properties list.
        nlag : int (default=2)
            An integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA
            sequence in the dataset). It represents the distance between two dinucleotides.
        all_property : bool (default=False)
            If True, returns all properties.
        extra_phyche_index : dict of str and list of float (default=None)
            The extra phyche index to use for the calculation. It means user-defined phyche_index. The key is
            dinucleotide, and its corresponding value is a list with a pair of physicochemical indices and its new
            value.
        Returns
        -------
        list of float
            The Dinucleotide Based Cross Covariance of the sequence.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_cc_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_DACC(self, phyche_index: List[str] = ["Twist", "Tilt"], nlag: int = 2, all_property: bool = False, extra_phyche_index: Dict[str, List[Tuple[str, float]]] = None) -> List[float]:
        """
        Calculates the Dinucleotide Based Auto Cross Covariance of the sequence. CODE FROM repDNA (https://github.com/liufule12/repDNA)
        Parameters
        ----------
        phyche_index : list of str, optional (default=["Twist", "Tilt"])
            The physicochemical properties list.
        nlag : int, optional (default=2)
            An integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA
            sequence in the dataset). It represents the distance between two dinucleotides.
        all_property : bool, optional (default=False)
            If True, returns all properties.
        extra_phyche_index : dict of str and list of float, optional (default=None)
            The extra phyche index to use for the calculation. It means user-defined phyche_index. The key is
            dinucleotide, and its corresponding value is a list with a pair of physicochemical indices and it's new
            value.
        Returns
        -------
        list of float
            The Dinucleotide Based Auto Cross Covariance of the sequence.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        zipped = list(zip(make_ac_vector([self.dna_sequence], nlag, phyche_value, k),
                          make_cc_vector([self.dna_sequence], nlag, phyche_value, k)))
        vector = [reduce(lambda x, y: x + y, e) for e in zipped]

        return vector[0]

    def get_TAC(self, phyche_index: List[str] = ["Dnase I", "Nucleosome"], nlag: int = 2, all_property: bool = False, extra_phyche_index: Dict[str, List[Tuple[str, float]]] = None) -> List[float]:
        """
        Calculates the Trinucleotide Based Auto Covariance of the sequence. CODE FROM repDNA (https://github.com/liufule12/repDNA)
        Parameters
        ----------
        phyche_index : list of str, optional (default=["Dnase I", "Nucleosome"])
            The physicochemical properties list.
        nlag : int, optional (default=3)
            An integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA
            sequence in the dataset). It represents the distance between two dinucleotides.
        all_property : bool, optional (default=False)
            If True, returns all properties.
        extra_phyche_index : dict of str and list of float, optional (default=None)
            The extra phyche index to use for the calculation. It means user-defined phyche_index. The key is
            trinucleotide, and its corresponding value is a list with a pair of physicochemical indices and it's new
            value.
        Returns
        -------
        list of float
            The Trinucleotide Based Auto Covariance of the sequence.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_ac_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_TCC(self, phyche_index: List[str] = ["Dnase I", "Nucleosome"], nlag: int = 2, all_property: bool = False, extra_phyche_index: Dict[str, List[Tuple[str, float]]] = None) -> List[float]:
        """
        Calculates the Trinucleotide Based Auto Covariance of the sequence. CODE FROM repDNA (https://github.com/liufule12/repDNA)
        Parameters
        ----------
        phyche_index : list of str, optional (default=None)
            The physicochemical properties list.
        nlag : int, optional (default=3)
            An integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA
            sequence in the dataset). It represents the distance between two dinucleotides.
        all_property : bool, optional (default=False)
            If True, returns all properties.
        extra_phyche_index : dict of str and list of float, optional (default=None)
            The extra phyche index to use for the calculation. It means user-defined phyche_index. The key is
            trinucleotide, and its corresponding value is a list with a pair of physicochemical indices and it's new
            value.
        Returns
        -------
        list of float
            The Trinucleotide Based Cross Covariance of the sequence.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)

        return make_cc_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_TACC(self, phyche_index: List[str] = ["Dnase I", "Nucleosome"], nlag: int = 2, all_property: bool = False, extra_phyche_index: Dict[str, List[Tuple[str, float]]] = None) -> List[float]:
        """
        Calculates the Dinucleotide Based Auto Cross Covariance of the sequence. CODE FROM repDNA (https://github.com/liufule12/repDNA)
        Parameters
        ----------
        phyche_index : list of str, optional (default=None)
            The physicochemical properties list.
        nlag : int, optional (default=2)
            An integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA
            sequence in the dataset). It represents the distance between two dinucleotides.
        all_property : bool, optional (default=False)
            If True, returns all properties.
        extra_phyche_index : dict of str and list of float, optional (default=None)
            The extra phyche index to use for the calculation. It means user-defined phyche_index. The key is
            trinucleotide, and its corresponding value is a list with a pair of physicochemical indices and it's new
            value.
        Returns
        -------
        list of float
            The Dinucleotide Based Auto Cross Covariance of the sequence.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)

        zipped = list(zip(make_ac_vector([self.dna_sequence], nlag, phyche_value, k),
                          make_cc_vector([self.dna_sequence], nlag, phyche_value, k)))
        vector = [reduce(lambda x, y: x + y, e) for e in zipped]

        return vector[0]

    # --------------------  PSEUDO NUCLEOTIDE COMPOSITION  -------------------- #

    def get_PseDNC(self, lamda: int = 3, w: float = 0.05) -> Dict[str, float]:
        """
        Calculates the Pseudo Dinucleotide Composition of the sequence. From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Parameters
        ----------
        lamda : int, optional (default=3)
            Value larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest sequence
            in the dataset). It represents the highest counted rank (or tier) of the correlation along a DNA sequence.
        w : float, optional (default=0.05)
            The weight factor ranged from 0 to 1.
        Returns
        -------
        Dict of str:float
            The Pseudo Dinucleotide Composition of the sequence.
        """
        d = {
            'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
            'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
            'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
            'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
            'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
            'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]
        }

        counts = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i + 2]
            counts[dinucleotide] += 1
            
        fk = {k: v / sum(counts.values()) for k, v in counts.items()}
        all_possibilites = make_kmer_list(2)

        thetas = []
        L = len(self.dna_sequence)
        for i in range(lamda):
            big_somatorio = 0.0
            for j in range(L-i-2):
                somatorio = 0.0
                first_dinucleotide = self.dna_sequence[j:j+2]
                second_dinucleotide = self.dna_sequence[j+i+1:j+i+1+2]
                for k in range(6):
                    val = (d[first_dinucleotide][k] -
                           d[second_dinucleotide][k])**2
                    somatorio += val

                big_somatorio += somatorio/6

            # Theta calculation
            if(L-i-2 == 0):
                theta = 0.0
            else:
                theta = big_somatorio / (L-i-2)
            thetas.append(theta)

        # --------------------------------------------

        res = {}
        for dinucleotide in all_possibilites:
            res[dinucleotide] = round(fk[dinucleotide] / (1 + w * sum(thetas)), 3)

        for i in range(lamda):
            res["lambda."+str(i+1)] = round(w * thetas[i] /
                                            (1 + w * sum(thetas)), 3)

        return res

    def get_PseKNC(self, k: int = 3, lamda: int = 1, w: float = 0.5) -> Dict[str, float]:
        """
        Calculates the Pseudo K Composition of the sequence. From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Parameters
        ----------
        k : int, optional (default=3)
            Value larger than 0 represents the k-tuple.
        lamda : int, optional (default=1)
            Value larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest sequence
            in the dataset). It represents the highest counted rank (or tier) of the correlation along a DNA sequence.
        w : float, optional (default=0.5)
            The weight factor ranged from 0 to 1.
        Returns
        -------
        Dict of str:float
            The Pseudo K Composition of the sequence.
        """
        d = {
            'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
            'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
            'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
            'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
            'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
            'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]
        }
        counts = make_kmer_dict(k)
        for i in range(len(self.dna_sequence) - k + 1):
            k_mer = self.dna_sequence[i:i + k]
            counts[k_mer] += 1
            
        fk = {k: v / sum(counts.values()) for k, v in counts.items()}
        all_possibilites: List[str] = make_kmer_list(k)

        thetas = []
        L = len(self.dna_sequence)
        for i in range(lamda):
            big_somatorio = 0.0
            for j in range(L-i-2):
                somatorio = 0.0
                first_dinucleotide = self.dna_sequence[j:j+2]
                second_dinucleotide = self.dna_sequence[j+i+1:j+i+1+2]
                for k in range(6):
                    val = (d[first_dinucleotide][k] - d[second_dinucleotide][k])**2
                    somatorio += val
                big_somatorio += somatorio/6

            # Theta calculation
            if(L-i-2 == 0):
                theta = 0.0
            else:
                theta = big_somatorio / (L-i-2)
            thetas.append(theta)

        # --------------------------------------------

        res = {}
        for k_tuple in all_possibilites:
            res[k_tuple] = round(fk[k_tuple] / (1 + w * sum(thetas)), 3)

        for i in range(lamda):
            res["lambda."+str(i+1)] = round(w * thetas[i] / (1 + w * sum(thetas)), 3)
        return res

    # ----------------------  CALCULATE DESCRIPTORS  ---------------------- #

    def get_descriptors(self, descriptor_list = []):
        """
        Calculates all descriptors
        Parameters
        ----------
        descriptor_list : List of str
            List of descriptors to be calculated the user wants to calculate. The list must be a subset of the
            descriptors in the list of descriptors. If the list is empty, all descriptors will be calculated.
        Returns
        -------
        Dict
            Dictionary with values of all descriptors
        """
        res = {}
        if(descriptor_list == []):
            res['length'] = self.get_length()
            res['gc_content'] = self.get_gc_content()
            res['at_content'] = self.get_at_content()
            res['nucleic_acid_composition'] = self.get_nucleic_acid_composition()
            res['dinucleotide_composition'] = self.get_dinucleotide_composition()
            res['trinucleotide_composition'] = self.get_trinucleotide_composition()
            res['k_spaced_nucleic_acid_pairs'] = self.get_k_spaced_nucleic_acid_pairs()
            res['kmer'] = self.get_kmer()
            res['accumulated_nucleotide_frequency'] = self.get_accumulated_nucleotide_frequency()
            res['DAC'] = self.get_DAC()
            res['DCC'] = self.get_DCC()
            res['DACC'] = self.get_DACC()
            res['TAC'] = self.get_TAC()
            res['TCC'] = self.get_TCC()
            res['TACC'] = self.get_TACC()
            res['PseDNC'] = self.get_PseDNC()
            res['PseKNC'] = self.get_PseKNC()
        else:
            for descriptor in descriptor_list:
                function = getattr(self, 'get_' + descriptor)
                res[descriptor] = function()
        return res
