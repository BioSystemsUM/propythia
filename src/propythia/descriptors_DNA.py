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

    def get_nucleic_acid_composition(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates nucleic acid composition
        :return: dictionary with values of nucleic acid composition
        """
        nucleic_acid_composition = {}
        for letter in self.dna_sequence:
            if letter in nucleic_acid_composition:
                nucleic_acid_composition[letter] += 1
            else:
                nucleic_acid_composition[letter] = 1
        return nucleic_acid_composition

    def get_gc_content(self):
        """
        Calculates gc content
        :return: value of gc content
        """
        gc_content = 0
        for letter in self.dna_sequence:
            print("letter:", letter)
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

    def get_dinucleotide_composition(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates dinucleotide composition
        :return: dictionary with values of dinucleotide composition
        """
        dinucleotide_composition = {}
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i+2]
            if dinucleotide in dinucleotide_composition:
                dinucleotide_composition[dinucleotide] += 1
            else:
                dinucleotide_composition[dinucleotide] = 1
        return dinucleotide_composition

    def get_trinucleotide_composition(self):
        """
        From: https://sci-hub.se/10.1093/bib/bbz041
        Calculates trinucleotide composition
        :return: dictionary with values of trinucleotide composition
        """
        trinucleotide_composition = {}
        for i in range(len(self.dna_sequence) - 2):
            trinucleotide = self.dna_sequence[i:i+3]
            if trinucleotide in trinucleotide_composition:
                trinucleotide_composition[trinucleotide] += 1
            else:
                trinucleotide_composition[trinucleotide] = 1
        return trinucleotide_composition

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
        reverse_complement = ""
        for letter in self.dna_sequence:
            reverse_complement += self.pairs[letter]
        return reverse_complement[::-1]
