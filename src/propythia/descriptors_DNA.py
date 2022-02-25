class DNADescriptor:

    pairs = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G'
    }

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

    def get_kmer(self, k=2, normalize=False, reverse=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://rdrr.io/cran/rDNAse/
        Calculates Kmer
        :param k: value of k
        :param normalize: default value is False. If True, this method returns the frequencies of all kmers.
        :param reverse: default value is False. If True, this method returns the reverse compliment kmer.
        :return: dictionary with values of kmer
        """
        res = {}

        for i in range(len(self.dna_sequence) - k + 1):
            if(self.dna_sequence[i:i+k] in res):
                res[self.dna_sequence[i:i+k]] += 1
            else:
                res[self.dna_sequence[i:i+k]] = 1

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

    # -----------------------  NUCLEIC ACID COMPOSITION ----------------------- #

    def get_nucleic_acid_composition(self, normalize=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates nucleic acid composition
        :param normalize: default value is False. If True, this method returns the frequencies of each nucleic acid.
        :return: dictionary with values of nucleic acid composition
        """
        res = {}
        for letter in self.dna_sequence:
            if letter in res:
                res[letter] += 1
            else:
                res[letter] = 1
        if normalize:
            N = sum(res.values())
            for key in res:
                res[key] = res[key] / N
        return res

    def get_dinucleotide_composition(self, normalize=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates dinucleotide composition
        :param normalize: default value is False. If True, this method returns the frequencies of each dinucleotide.
        :return: dictionary with values of dinucleotide composition
        """
        res = {}
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i+2]
            if dinucleotide in res:
                res[dinucleotide] += 1
            else:
                res[dinucleotide] = 1
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
        res = {}
        for i in range(len(self.dna_sequence) - 2):
            trinucleotide = self.dna_sequence[i:i+3]
            if trinucleotide in res:
                res[trinucleotide] += 1
            else:
                res[trinucleotide] = 1
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

        :return: dictionary with values of nucleotide chemical property
        """
        chemical_property = {
            'A': [1, 1, 1],
            'C': [0, 1, 0],
            'G': [1, 0, 0],
            'T': [0, 0, 1],
        }
        return [chemical_property[i] for i in self.dna_sequence]

    # --------------------  PSEUDO NUCLEOTIDE COMPOSITION  -------------------- #

    def get_pseudo_dinucleotide_composition(self):
        pass

    def get_pseudo_k_tupler_composition(self):
        pass

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
        res['kmer'] = self.get_kmer()
        res['nucleic_acid_composition'] = self.get_nucleic_acid_composition()
        res['dinucleotide_composition'] = self.get_dinucleotide_composition()
        res['trinucleotide_composition'] = self.get_trinucleotide_composition()
        res['nucleotide_chemical_property'] = self.get_nucleotide_chemical_property()
        return res
