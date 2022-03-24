"""
##############################################################################

A class  used for computing different types of protein encoddings parallelized.
It contains encodings such nfl, blossum, zcale, one-hot-encoding. It also allows to perform padding.

Authors: Miguel Barros

Date: 03/2022

Email:

##############################################################################
"""
import os
import pandas as pd
from joblib import Parallel, delayed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class Encoding:
    def __init__(self, dataset, col: str = 'sequence'):
        """
        Constructor

        :param dataset: the data corresponding to the protein sequences, it should be an string (one sequence),
         a list  or a pandas dataframe (multiples sequences).
        :param col: the name of the column in the dataframe which contains the protein sequences (pandas dataframe),
        or the name to give to the protein sequence column (list or string). Default collumn name is 'sequence'.
        """
        if isinstance(dataset, pd.DataFrame):
            self.result = dataset
        elif isinstance(dataset, str):
            data = {col: [dataset]}
            self.result = pd.DataFrame(data)
        elif isinstance(dataset, list):
            data = {col: dataset}
            self.result = pd.DataFrame(data)
        else:
            raise Exception('Parameter dataframe is not an string, list or pandas Dataframe')

        self.col = col
        self.padded = False
        self.result.drop_duplicates(subset = self.col, keep='first', inplace=True)

        for path in [os.path.split(__file__)[0]]:
            if os.path.exists(os.path.join(path, str(''))):
                self.path = path
                break

    def get_seq_pad(self, seq_len: int = 600, alphabet: str = "XARNDCEQGHILKMFPSTWYV", padding_truncating='post',
                    n_jobs: int = 4):
        """
       This methods performs the padding of the sequences.

       :param seq_len: The maximum length for all sequences. By default the length is 600 amino acids
       :param alphabet: The alphabet of aminoacids to be used.
       :param padding_truncating: 'pre' or 'post' pad either before or after each sequence, also removes values from sequences larger than seq_len. By default padding is after each sequence.
       :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores

       :return: Dataframe with the padded sequences for each sequence.
       """

        if alphabet[0] != 'X':
            alphabet = alphabet.replace('X', '')
            alphabet = 'X' + alphabet
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))  # dicionarios de hot encoding
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))  # dicionarios de reversão

        try:
            with Parallel(n_jobs=n_jobs) as parallel:
                res = parallel(
                    delayed(seq_padding)(seq, self.col, char_to_int, int_to_char, seq_len, padding_truncating) for seq
                    in self.result[self.col])
            res = pd.DataFrame(res)
        except:
            raise Exception('Sequences not preprocessed. Run sequence preprocessing')
        return res

    def get_hot_encoded(self, alphabet: str = "XARNDCEQGHILKMFPSTWYV", n_jobs: int = 4):
        """
        This methods performs the one-hot-encoding of sequences.

        :param alphabet: The alphabet of aminoacids to be used.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores

        :return: Dataframe with the one-hot-encoding for each sequence
        """

        if alphabet[0] != 'X':
            alphabet = alphabet.replace('X', '')
            alphabet = 'X' + alphabet
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))  # dicionario de hot encoding
        try:
            with Parallel(n_jobs=n_jobs) as parallel:
                res = parallel(delayed(seq_hot_encoded)(seq, self.col, char_to_int) for seq in self.result[self.col])
            res = pd.DataFrame(res)
            if self.padded: res.drop_duplicates(subset='padded_sequence', keep='first', inplace=True)
            return res
        except:
            raise Exception('Amino acids not included in the alphabet found. Please run preprocessing methods.')

    def get_pad_and_hot_encoding(self, seq_len: int = 600, alphabet: str = "XARNDCEQGHILKMFPSTWYV",
                                 padding_truncating='post', n_jobs: int = 4):
        """
        This methods performs the padding of the sequences and one-hot-encode the padded sequences.

        :param seq_len: The maximum length for all sequences. By default the length is 600 amino acids
        :param alphabet: The alphabet of aminoacids to be used.
        :param padding_truncating: 'pre' or 'post' pad either before or after each sequence, also removes values from sequences larger than seq_len. By default padding is after each sequence.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores

        :return: Dataframe with the padded sequences and the one-hot-encoding for each sequence
        """
        if alphabet[0] != 'X':
            alphabet = alphabet.replace('X', '')
            alphabet = 'X' + alphabet
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))  # dicionarios de hot encoding
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))  # dicionarios de reversão
        try:
            with Parallel(n_jobs=n_jobs) as parallel:
                res = parallel(delayed(seq_padded_hot)(seq, self.col, char_to_int, int_to_char, seq_len,
                                                       padding_truncating) for seq in self.result[self.col])
            res = pd.DataFrame(res)
            return res
        except:
            raise Exception('Amino acids not included in the alphabet found. Please run preprocessing methods.')

    def get_blosum(self, blosum: str = 'blosum62', n_jobs: int = 4):
        """
        BLOSUM62 is a substitution matrix that specifies the similarity of one amino acid to another by means of a score.
        This score reflects the frequency of substitutions found from studying protein sequence conservation
        in large databases of related proteins.
        The number 62 refers to the percentage identity at which sequences are clustered in the analysis.
        I is possible to get blosum50 to get 50 % identity.
        Encoding a peptide this way means we provide the column from the blosum matrix corresponding to the amino acid
        at each position of the sequence. This produces 24 * seqlen matrix.

        :param blosum: blosum matrix to use either 'blosum62' or 'blosum50'. by default 'blosum62'
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores

        :return: Dataframe with blosum encoding for each sequence
        """

        if blosum == 'blosum62':
            encoding = pd.read_csv(self.path + '/adjuv_functions/features_functions/data/blosum62.csv',
                                   index_col=0).to_dict()
        elif blosum == 'blosum50':
            encoding = pd.read_csv(self.path + '/adjuv_functions/features_functions/data/blosum50.csv',
                                   index_col=0).to_dict()
        else:
            raise Exception('The provided encoding is not valid')
        try:
            if isinstance(encoding, dict):
                with Parallel(n_jobs=n_jobs) as parallel:
                    res = parallel(delayed(seq_blosum_encoding)(seq, self.col, encoding) for seq in self.result[self.col])
                res = pd.DataFrame(res)
                if self.padded: res.drop_duplicates(subset='padded_sequence', keep='first', inplace=True)
                return res
            else:
                raise Exception('The provided encoding is not a dictionary')
        except:
            raise Exception('Sequences not preprocessed. Run sequence preprocessing')

    def get_nlf(self, n_jobs: int = 4):
        """
        Method that takes many physicochemical properties and transforms them using a Fisher Transform (similar to a PCA)
        creating a smaller set of features that can describe the amino acid just as well.
        There are 19 transformed features.
        This method of encoding is detailed by Nanni and Lumini in their paper:
        L. Nanni and A. Lumini, “A new encoding technique for peptide classification,”
        Expert Syst. Appl., vol. 38, no. 4, pp. 3185–3191, 2011
        This function just receives 20aa letters, therefore preprocessing is required.

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the Fisher Transform encoding for each sequence.
        """
        encoding = pd.read_csv(self.path + '/adjuv_functions/features_functions/data/nlf.csv', index_col=0).to_dict()
        try:
            if isinstance(encoding, dict):
                with Parallel(n_jobs=n_jobs) as parallel:
                    res = parallel(delayed(seq_nlf_encoding)(seq, self.col, encoding) for seq in self.result[self.col])
                res = pd.DataFrame(res)
                if self.padded: res.drop_duplicates(subset='padded_sequence', keep='first', inplace=True)
                return res
            else:
                raise Exception('The provided encoding is not valid')
        except:
            raise Exception('Sequences not preprocessed. Run sequence preprocessing')

    def get_zscale(self, n_jobs: int = 4):
        """
        This method encodes which amino acid of the sequence into a Z-scales. Each Z scale represent an amino-acid property:
        Z1: Lipophilicity
        Z2: Steric properties (Steric bulk/Polarizability)
        Z3: Electronic properties (Polarity / Charge)
        Z4 and Z5: They relate electronegativity, heat of formation, electrophilicity and hardness.

        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with the Z-scale encoding for each sequence.
        """
        encoding = zs
        try:
            if isinstance(encoding, dict):
                with Parallel(n_jobs=n_jobs) as parallel:
                    res = parallel(delayed(seq_zscale_encoding)(seq, self.col, encoding) for seq in self.result[self.col])
                res = pd.DataFrame(res)
                if self.padded: res.drop_duplicates(subset='padded_sequence', keep='first', inplace=True)
                return res
            else:
                raise Exception('The provided encoding is not valid')
        except:
            raise Exception('Sequences not preprocessed. Run sequence preprocessing')

    def get_all(self, alphabet: str = "XARNDCEQGHILKMFPSTWYV", seq_len: int = 600, padding_truncating='post',
                blosum: str = 'blosum62', use_padded: bool = False, n_jobs: int = 4):
        """
        Calculate all encoding functions for the protein sequences.

        :param alphabet: The alphabet of aminoacids to be used.
        :param seq_len: The maximum length for all sequences. By default the length is 600 amino acids
        :param padding_truncating: 'pre' or 'post' pad either before or after each sequence, also removes values from sequences larger than seq_len. By default padding is after each sequence.
        :param blosum: blosum matrix to use either 'blosum62' or 'blosum50'. by default 'blosum62'
        :param use_padded: A bool argument, if true it uses the padded sequences for the one-hot, blosum and z-scale encoding.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with all encodings for each sequence.
        """
        if use_padded:
            self.result = self.result.merge(self.get_pad_and_hot_encoding(seq_len, alphabet, padding_truncating, n_jobs), how='left',
                                            on=self.col)
            self.col = 'padded_sequence'
            self.padded = True
        else:
            self.result = self.result.merge(self.get_seq_pad(seq_len, alphabet, padding_truncating, n_jobs), how='left', on=self.col)
            self.result = self.result.merge(self.get_hot_encoded(alphabet, n_jobs), how='left', on=self.col)
        self.result = self.result.merge(self.get_blosum(blosum, n_jobs), how='left', on=self.col)
        self.result = self.result.merge(self.get_nlf(n_jobs), how='left', on=self.col)
        self.result = self.result.merge(self.get_zscale(n_jobs), how='left', on=self.col)

        return self.result

    def get_adaptable(self, list_of_functions : list, alphabet: str = "XARNDCEQGHILKMFPSTWYV", seq_len: int = 600, padding_truncating='post',
                blosum: str = 'blosum62', use_padded: bool = False, n_jobs: int = 4):
        """
        It allows to run a selected set of encoding functions for the protein sequences.

        :param list_of_functions: A list with the fuctions to run.
        :param alphabet: The alphabet of aminoacids to be used.
        :param seq_len: The maximum length for all sequences. By default the length is 600 amino acids
        :param padding_truncating: 'pre' or 'post' pad either before or after each sequence, also removes values from sequences larger than seq_len. By default padding is after each sequence.
        :param blosum: blosum matrix to use either 'blosum62' or 'blosum50'. by default 'blosum62'
        :param use_padded: A bool argument, if true it uses the padded sequences for the one-hot, blosum and z-scale encoding.
        :param n_jobs: number of CPU cores to be used. Default used is 4 CPU cores
        :return: Dataframe with all encodings for each sequence.
        """

        for function in list_of_functions:
            if function == 1:
                self.result = self.result.merge(self.get_seq_pad(seq_len, alphabet, padding_truncating, n_jobs), how='left',on=self.col)
                if use_padded: self.col = 'padded_sequence'
            if function == 2: self.result  = self.result.merge(self.get_hot_encoded(alphabet, n_jobs), how='left', on=self.col)
            if function == 3: self.result = self.result.merge(self.get_pad_and_hot_encoding(seq_len, alphabet, padding_truncating, n_jobs), how='left',on=self.col)
            if function == 4: self.result = self.result.merge(self.get_blosum(blosum, n_jobs), how='left', on=self.col)
            if function == 5: self.result = self.result.merge(self.get_nlf(n_jobs), how='left', on=self.col)
            if function == 6: self.result = self.result.merge(self.get_zscale(n_jobs), how='left', on=self.col)
            if function == 7: self.get_all(alphabet, seq_len, padding_truncating, blosum, use_padded, n_jobs)

        return self.result

def seq_blosum_encoding(ProteinSequence: str, col: str, encoding: dict):
    """
    BLOSUM62 is a substitution matrix that specifies the similarity of one amino acid to another by means of a score.
    This score reflects the frequency of substitutions found from studying protein sequence conservation
    in large databases of related proteins.
    The number 62 refers to the percentage identity at which sequences are clustered in the analysis.
    I is possible to get blosum50 to get 50 % identity.
    Encoding a peptide this way means we provide the column from the blosum matrix corresponding to the amino acid
    at each position of the sequence. This produces 24*seqlen matrix.

    :param ProteinSequence: protein sequence to be encoded
    :param col: name of the collum for the original protein sequences.
    :param encoding: dict with the encoding. Keys are the aminoacid residues and values the encoding.
    :return: dict form with blosum encoding
    """
    res = {col: ProteinSequence}
    res['blosum'] = [list(encoding[i].values()) for i in ProteinSequence]
    return res

def seq_nlf_encoding(ProteinSequence: str, col: str, encoding : dict):
    """
    Method that takes many physicochemical properties and transforms them using a Fisher Transform (similar to a PCA)
    creating a smaller set of features that can describe the amino acid just as well.
    There are 19 transformed features.
    This method of encoding is detailed by Nanni and Lumini in their paper:
    L. Nanni and A. Lumini, “A new encoding technique for peptide classification,”
    Expert Syst. Appl., vol. 38, no. 4, pp. 3185–3191, 2011
    This function just receives 20aa letters, therefore preprocessing is required.

    :param ProteinSequence: protein sequence to be encoded
    :param col: name of the collum for the original protein sequences.
    :param encoding: dict with the encoding. Keys are the aminoacid residues and values the encoding.
    :return: Dict form with the Fisher Transform encoding for the sequence.
    """
    res = {col: ProteinSequence}
    ProteinSequence = ProteinSequence.replace('X','')
    res['nlf'] = [list(encoding[i].values()) for i in ProteinSequence]
    return res


def seq_zscale_encoding(ProteinSequence : str, col : str, encoding):
    """
    This method encodes which amino acid of the sequence into a Z-scales. Each Z scale represent an amino-acid property:
    Z1: Lipophilicity
    Z2: Steric properties (Steric bulk/Polarizability)
    Z3: Electronic properties (Polarity / Charge)
    Z4 and Z5: They relate electronegativity, heat of formation, electrophilicity and hardness.

    :param ProteinSequence: protein sequence to be encoded
    :param col: name of the collum for the original protein sequences.
    :param encoding: dict with the encoding. Keys are the aminoacid residues and values the encoding.
    :return: dict form with the Z-scale encoding for the sequence.
    """
    res = {col: ProteinSequence}
    res['zscale'] = [encoding[i] for i in ProteinSequence]
    return res


def seq_padding(ProteinSequence : str, col: str, char_to_int: dict, int_to_char: dict, seq_len: int,
                padding_truncating: str = 'post'):
    """
    This methods performs the padding of the sequence.

    :param ProteinSequence: protein sequence to be encoded
    :param col: name of the collum for the original protein sequences.
    :param char_to_int: dict with the encoding. Keys are the aminoacid residues and values the corresponding int.
    :param int_to_char: dict with the encoding. Keys are the corresponding int and the the values aminoacid residues.
    :param seq_len: The maximum length for all sequences. By default the length is 600 amino acids.
    :param padding_truncating: 'pre' or 'post' pad either before or after each sequence, also removes values from sequences larger than seq_len. By default padding is after each sequence.

    :return: Dict form with the padded sequences for the sequence.
    """
    integer_encoded = [[char_to_int[char] for char in ProteinSequence]]
    list_of_sequences_length = pad_sequences(integer_encoded, maxlen=seq_len, dtype='int32',
                                             padding=padding_truncating, truncating=padding_truncating, value=0)

    char_paded = [int_to_char[i] for i in list_of_sequences_length[0]]
    pad_aa = ''.join(char_paded)
    res = {col: ProteinSequence}
    res['padded_sequence'] = pad_aa
    return res


def seq_hot_encoded(ProteinSequence : str, col: str, char_to_int: dict):
    """
    This methods performs the one-hot-encoding of the sequence.

    :param ProteinSequence: protein sequence to be encoded
    :param col: name of the collum for the original protein sequences.
    :param char_to_int: dict with the encoding. Keys are the aminoacid residues and values the corresponding int.

    :return: Dict form with the one-hot-encoding for the sequence.
    """
    integer_encoded = [char_to_int[char] for char in ProteinSequence]

    hot_enc = to_categorical(integer_encoded).tolist()
    res = {col : ProteinSequence}
    res['One_hot_encoding'] = hot_enc
    return res

def seq_padded_hot(ProteinSequence : str, col: str, char_to_int: dict, int_to_char: dict, seq_len: int,
                padding_truncating: str = 'post'):
    """
    This methods performs the padding of the sequence and one-hot-encode the padded sequence.

    :param ProteinSequence: protein sequence to be encoded
    :param col: name of the collum for the original protein sequences.
    :param char_to_int: dict with the encoding. Keys are the aminoacid residues and values the corresponding int.
    :param int_to_char: dict with the encoding. Keys are the corresponding int and the the values aminoacid residues.
    :param seq_len: The maximum length for all sequences. By default the length is 600 amino acids.
    :param padding_truncating: 'pre' or 'post' pad either before or after each sequence, also removes values from sequences larger than seq_len. By default padding is after each sequence.

    :return: Dict form with the padded sequences and the one-hot-encoding for the sequence.
    """
    integer_encoded = [char_to_int[char] for char in ProteinSequence]
    list_of_sequences_length = pad_sequences([integer_encoded], maxlen=seq_len, dtype='int32',
                                             padding=padding_truncating, truncating=padding_truncating, value=0)

    char_paded = [int_to_char[i] for i in list_of_sequences_length[0]]
    pad_aa = ''.join(char_paded)

    hot_enc = to_categorical(list_of_sequences_length[0]).tolist()

    res = {col: ProteinSequence}
    res['padded_sequence'] = pad_aa
    res['One_hot_encoding'] = hot_enc

    return res

zs = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
    'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
    'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
    'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
    'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
    'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
    'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
    'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
    'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
    'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
    'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
    'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
    'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
    'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
    'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
    'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
    'X': [0.0, 0.0, 0.0, 0.0, 0.0]}
