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
        From: https://sci-hub.se/10.1093/bib/bbz041
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

    def get_kmer(self, k, reverse=False):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates Kmer
        :param k: value of k
        :param reverse: default value is False. If True, this method returns the reverse compliment kmer.
        :return: dictionary with values of kmer
        """
        kmer = {}
        new_sequence = self.dna_sequence if not reverse else self.get_reverse_complement()
        for i in range(len(new_sequence) - k + 1):
            if(new_sequence[i:i+k] in kmer):
                kmer[new_sequence[i:i+k]] += 1
            else:
                kmer[new_sequence[i:i+k]] = 1
        return kmer

    def get_reverse_complement(self):
        """
        Calculates reverse complement (Auxiliary function to "get_kmer")
        :return: reverse complement of sequence
        """
        res = ""
        for letter in self.dna_sequence:
            res += self.pairs[letter]
        return res[::-1]

    # -----------------------  NUCLEIC ACID COMPOSITION ----------------------- #

    def get_nucleic_acid_composition(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates nucleic acid composition
        :return: dictionary with values of nucleic acid composition
        """
        res = {}
        for letter in self.dna_sequence:
            if letter in res:
                res[letter] += 1
            else:
                res[letter] = 1
        return res

    def get_dinucleotide_composition(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates dinucleotide composition
        :return: dictionary with values of dinucleotide composition
        """
        res = {}
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i+2]
            if dinucleotide in res:
                res[dinucleotide] += 1
            else:
                res[dinucleotide] = 1
        return res

    def get_trinucleotide_composition(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates trinucleotide composition
        :return: dictionary with values of trinucleotide composition
        """
        res = {}
        for i in range(len(self.dna_sequence) - 2):
            trinucleotide = self.dna_sequence[i:i+3]
            if trinucleotide in res:
                res[trinucleotide] += 1
            else:
                res[trinucleotide] = 1
        return res

    def get_accumulated_nucleotide_frequency(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates accumulated nucleotide frequency
        :return: dictionary with values of accumulated nucleotide frequency in percentage
        """
        res = {}
        for letter in self.dna_sequence:
            if letter in res:
                res[letter] += 1
            else:
                res[letter] = 1
        for key in res:
            res[key] = res[key] / self.get_length()
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
        res['kmer'] = self.get_kmer(k=3)
        res['nucleic_acid_composition'] = self.get_nucleic_acid_composition()
        res['dinucleotide_composition'] = self.get_dinucleotide_composition()
        res['trinucleotide_composition'] = self.get_trinucleotide_composition()
        res['accumulated_nucleotide_frequency'] = self.get_accumulated_nucleotide_frequency()
        res['nucleotide_chemical_property'] = self.get_nucleotide_chemical_property()
        return res
