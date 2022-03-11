# -*- coding: utf-8 -*-
"""
##############################################################################

A class used for computing different types of DNA descriptors.
It contains descriptors from packages iLearn, iDNA4mC, rDNAse, ...

Authors: João Nuno Abreu

Date: 02/2021

Email:

##############################################################################
"""


from utils import make_kmer_list, make_kmer_dict, ready_acc, make_ac_vector, make_cc_vector
from functools import reduce


class DNADescriptor:

    pairs = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G'
    }

    ALPHABET = 'ACGT'

    """
    The Descriptor class collects all descriptor calculation functions into a simple class.
    It returns the features in a dictionary object
    """

    def __init__(self, dna_sequence):
        """	Constructor """
        self.dna_sequence = dna_sequence.strip().upper()

    def get_length(self):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates lenght of sequence (number of aa)
        :return: value of length
        """
        return len(self.dna_sequence)

    def get_gc_content(self):
        """
        Calculates gc content
        :return: value of gc content
        """
        gc_content = 0
        for letter in self.dna_sequence:
            if letter == 'G' or letter == 'C':
                gc_content += 1
        return gc_content / self.get_length()

    def get_at_content(self):
        """
        Calculates at content
        :return: value of at content
        """
        at_content = 0
        for letter in self.dna_sequence:
            if letter == 'A' or letter == 'T':
                at_content += 1
        return at_content / self.get_length()

    # ----------------------- NUCLEIC ACID COMPOSITION ----------------------- #

    def get_nucleic_acid_composition(self, normalize=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates nucleic acid composition
        :param normalize: default value is False. If True, this method returns the frequencies of each nucleic acid.
        :return: dictionary with values of nucleic acid composition
        """
        res = make_kmer_dict(1)
        for letter in self.dna_sequence:
            res[letter] += 1

        if normalize:
            N = sum(res.values())
            for key in res:
                res[key] = res[key] / N
        return res

    def get_enhanced_nucleic_acid_composition(self, window_size=5, normalize=False):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6216033/#SM0, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates enhanced nucleic acid composition
        :param normalize: default value is False. If True, this method returns the frequencies of each nucleic acid.
        :return: dictionary with values of enhanced nucleic acid composition
        """
        res = []
        for i in range(len(self.dna_sequence) - window_size + 1):
            aux_d = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            segment = self.dna_sequence[i:i+window_size]

            for letter in segment:
                aux_d[letter] += 1

            if normalize:
                N = sum(aux_d.values())
                for key in aux_d:
                    aux_d[key] = aux_d[key] / N

            res.append(aux_d)

        return res

    def get_dinucleotide_composition(self, normalize=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates dinucleotide composition
        :param normalize: default value is False. If True, this method returns the frequencies of each dinucleotide.
        :return: dictionary with values of dinucleotide composition
        """
        res = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i+2]
            res[dinucleotide] += 1
        if normalize:
            N = sum(res.values())
            for key in res:
                res[key] = res[key] / N
        return res

    def get_trinucleotide_composition(self, normalize=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates trinucleotide composition
        :param normalize: default value is False. If True, this method returns the frequencies of each trinucleotide.
        :return: dictionary with values of trinucleotide composition
        """
        res = make_kmer_dict(3)
        for i in range(len(self.dna_sequence) - 2):
            trinucleotide = self.dna_sequence[i:i+3]
            res[trinucleotide] += 1

        if normalize:
            N = sum(res.values())
            for key in res:
                res[key] = res[key] / N
        return res

    def get_k_spaced_nucleic_acid_pairs(self, k=0, normalize=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates k-spaced nucleic acid pairs
        :param k: value of k
        :param normalize: default value is False. If True, this method returns the frequencies of each k-spaced nucleic acid pair.
        :return: dictionary with values of k-spaced nucleic acid pairs
        """
        res = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - k - 1):
            k_spaced_nucleic_acid_pair = self.dna_sequence[i] + \
                self.dna_sequence[i+k+1]
            res[k_spaced_nucleic_acid_pair] += 1

        if normalize:
            N = sum(res.values())
            for key in res:
                res[key] = res[key] / N
        return res

    def get_kmer(self, k=2, normalize=False, reverse=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://rdrr.io/cran/rDNAse/
        Calculates Kmer
        :param k: value of k
        :param normalize: default value is False. If True, this method returns the frequencies of all kmers.
        :param reverse: default value is False. If True, this method returns the reverse compliment kmer.
        :return: dictionary with values of kmer
        """
        res = make_kmer_dict(k)

        for i in range(len(self.dna_sequence) - k + 1):
            res[self.dna_sequence[i:i+k]] += 1

        if reverse:
            for kmer, _ in sorted(res.items(), key=lambda x: x[0]):
                reverse = "".join([self.pairs[i] for i in kmer[::-1]])

                # calculate alphabet order between kmer and reverse compliment
                if(kmer < reverse):
                    smaller = kmer
                    bigger = reverse
                else:
                    smaller = reverse
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
            N = sum(res.values())
            for key in res:
                res[key] = res[key] / N

        return res

    def get_nucleotide_chemical_property(self):
        """
        From: https://academic.oup.com/bioinformatics/article/33/22/3518/4036387

        Calculates nucleotide chemical property

        Chemical property | Class	   | Nucleotides
        -------------------------------------------
        Ring structure 	  | Purine 	   | A, G
                          | Pyrimidine | C, T
        -------------------------------------------
        Hydrogen bond 	  | Strong 	   | C, G
                          | Weak 	   | A, T
        -------------------------------------------
        Functional group  | Amino 	   | A, C
                          | Keto 	   | G, T

        :return: list with values of nucleotide chemical property
        """
        chemical_property = {
            'A': [1, 1, 1],
            'C': [0, 1, 0],
            'G': [1, 0, 0],
            'T': [0, 0, 1],
        }
        return [chemical_property[i] for i in self.dna_sequence]

    def get_accumulated_nucleotide_frequency(self):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://www.nature.com/articles/srep13859?proof=t%252Btarget%253D
        Calculates accumulated nucleotide frequency
        :return: list with values of accumulated nucleotide frequency
        """
        res = []
        aux_d = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for i in range(len(self.dna_sequence)):
            aux_d[self.dna_sequence[i]] += 1
            x = aux_d[self.dna_sequence[i]] / (i + 1)
            res.append(x)
        return res

    # -------------------------------  Binary  ------------------------------ #

    def get_binary(self):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates binary encoding. Each nucleotide is encoded by a four digit binary vector.
        :return: list with values of binary encoding
        """
        binary = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]
        }
        return [binary[i] for i in self.dna_sequence]

    # --------------------------  Autocorrelation  -------------------------- #

    def get_DAC(self, phyche_index=["Twist", "Tilt"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make DAC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_ac_vector([self.dna_sequence], nlag, phyche_value, k)

    def get_DCC(self, phyche_index=["Twist", "Tilt"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make DCC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_cc_vector([self.dna_sequence], nlag, phyche_value, k)

    def get_DACC(self, phyche_index=["Twist", "Tilt"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make DACC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        zipped = list(zip(make_ac_vector([self.dna_sequence], nlag, phyche_value, k),
                          make_cc_vector([self.dna_sequence], nlag, phyche_value, k)))
        vector = [reduce(lambda x, y: x + y, e) for e in zipped]

        return vector

    def get_TAC(self, phyche_index=["Dnase I", "Nucleosome"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make TAC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_ac_vector([self.dna_sequence], nlag, phyche_value, k)

    def get_TCC(self, phyche_index=["Dnase I", "Nucleosome"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make TCC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)

        return make_cc_vector([self.dna_sequence], nlag, phyche_value, k)

    def get_TACC(self, phyche_index=["Dnase I", "Nucleosome"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make get_TACC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)

        zipped = list(zip(make_ac_vector([self.dna_sequence], nlag, phyche_value, k),
                          make_cc_vector([self.dna_sequence], nlag, phyche_value, k)))
        vector = [reduce(lambda x, y: x + y, e) for e in zipped]

        return vector

    # --------------------  PSEUDO NUCLEOTIDE COMPOSITION  -------------------- #

    def get_PseDNC(self, lamda=3, w=0.05):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates the Pseudo Dinucleotide Composition Descriptor of DNA sequences
        :param lamda: value of lambda
        :param w: value of w
        :return: dictionary with values of Pseudo Dinucleotide Composition Descriptor
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

        fk = self.get_dinucleotide_composition(normalize=True)
        all_possibilites = make_kmer_list(2)

        thetas = []
        L = len(self.dna_sequence)
        for i in range(lamda):
            big_somatorio = 0
            for j in range(L-i-2):
                somatorio = 0
                first_dinucleotide = self.dna_sequence[j:j+2]
                second_dinucleotide = self.dna_sequence[j+i+1:j+i+1+2]
                for k in range(6):
                    val = (d[first_dinucleotide][k] -
                           d[second_dinucleotide][k])**2
                    somatorio += val

                big_somatorio += somatorio/6

            # Theta calculation
            theta = big_somatorio / (L-2-i)
            thetas.append(theta)

        # --------------------------------------------

        res = {}
        for i in all_possibilites:
            res[i] = round(fk[i] / (1 + w * sum(thetas)), 3)

        for i in range(lamda):
            res["lambda."+str(i+1)] = round(w * thetas[i] /
                                            (1 + w * sum(thetas)), 3)

        return res

    def get_PseKNC(self, k=3, lamda=1, w=0.5):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates the Pseudo K Composition Descriptor of DNA sequences
        :param lamda: value of lambda
        :param w: value of w
        :return: dictionary with values of Pseudo K Composition Descriptor
        """
        d = {
            'AA': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
            'AC': [1.50, 0.50, 1.19, 0.13, 1.29, 1.04],
            'AG': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
            'AT': [1.07, 0.22, 0.83, -1.02, 2.51, 1.17],
            'CA': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
            'CC': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
            'CG': [-1.66, -1.22, -1.38, -0.82, -0.29, -1.39],
            'CT': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
            'GA': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
            'GC': [-0.08, 0.22, 2.3, -0.35, 0.65, 1.59],
            'GG': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
            'GT': [1.50, 0.50, 1.19, 0.13, 1.29, 1.04],
            'TA': [-1.23, -2.37, -1.38, -2.24, -1.51, -1.39],
            'TC': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
            'TG': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
            'TT': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11]
        }
        fk = self.get_kmer(k=k, normalize=True)
        all_possibilites = make_kmer_list(k)

        thetas = []
        L = len(self.dna_sequence)
        for i in range(lamda):
            big_somatorio = 0
            for j in range(L-lamda-1):
                somatorio = 0
                first_dinucleotide = self.dna_sequence[j:j+2]
                second_dinucleotide = self.dna_sequence[j+i+1:j+i+1+2]
                for k in range(6):
                    val = (d[first_dinucleotide][k] -
                           d[second_dinucleotide][k])**2
                    somatorio += val
                big_somatorio += somatorio/6

            # Theta calculation
            theta = big_somatorio / (L-i-2)
            thetas.append(theta)

        # --------------------------------------------

        res = {}
        for i in all_possibilites:
            res[i] = round(fk[i] / (1 + w * sum(thetas)), 3)

        for i in range(lamda):
            res["lambda."+str(i+1)] = round(w * thetas[i] /
                                            (1 + w * sum(thetas)), 3)
        return res

    # ----------------------  CALCULATE ALL DESCRIPTORS  ---------------------- #

    def get_all_descriptors(self):
        """
        Calculates all descriptors
        :return: dictionary with values of all descriptors
        """
        res = {}
        res['length'] = self.get_length()
        res['gc_content'] = self.get_gc_content()
        res['at_content'] = self.get_at_content()
        res['nucleic_acid_composition'] = self.get_nucleic_acid_composition()
        res['enhanced_nucleic_acid_composition'] = self.get_enhanced_nucleic_acid_composition()
        res['dinucleotide_composition'] = self.get_dinucleotide_composition()
        res['trinucleotide_composition'] = self.get_trinucleotide_composition()
        res['k_spaced_nucleic_acid_pairs'] = self.get_k_spaced_nucleic_acid_pairs()
        res['kmer'] = self.get_kmer()
        res['nucleotide_chemical_property'] = self.get_nucleotide_chemical_property()
        res['accumulated_nucleotide_frequency'] = self.get_accumulated_nucleotide_frequency()
        res['binary'] = self.get_binary()
        res['DAC'] = self.get_DAC()
        res['DCC'] = self.get_DCC()
        res['DACC'] = self.get_DACC()
        res['TAC'] = self.get_TAC()
        res['TCC'] = self.get_TCC()
        res['TACC'] = self.get_TACC()
        res['PseDNC'] = self.get_PseDNC()
        res['PseKNC'] = self.get_PseKNC()
        return res
