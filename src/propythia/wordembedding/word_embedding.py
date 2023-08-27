
import json
import csv
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import tensorflow as tf
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec
pd.options.mode.chained_assignment = None
import numpy as np
import os
from gensim.models import KeyedVectors
#from sklearn.feature_selection import VarianceThreshold

# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')
#
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class WordEmbedding:
    """
    Class for train and save a word embedding. Load pre-trained embedding model and create embedding vectors.
    """
    def __init__(self , model_file= None,
                 overlapping = True,
                 w2v:int = 1 ,
                 sg:int = 1 ,
                 vectordim: int = 100 ,
                 ngram_len: int = 3 ,
                 AA_substitute: str = True,
                 sequence_max_len: int = None,
                 emb_matrix_file: str= None ,
                 windowsize = 5 ,
                 filename = None):
        """
        Construct a new 'WordEmbedding' object.
        :param model_file: Pre-trained embedding model
        :param overlooping: True = Ngram segmentation using overlaping. False = Ngram segmentation using non-overlaping
        :param w2v:Training architecture: 1 for word2vec; otherwise fastText.
        :param sg:Training algorithm: 1 for skip-gram; otherwise CBOW.
        :param vectordim: Dimensionality of the word vectors.
        :param ngram_len: Length of the ngram
        :param AA_substitute: True = remove all of X amino acids and replace all amino acids B, Z, U and O for N, Q, C and Q.
        :param sequence_max_len: Maximum length of the biological sequence
        :param emb_matrix_file: Pre-trained embedding matrix in a .csv file
        :param windowsize: Maximum distance between the current and predicted word within a sentence
        :param filename: Given name to any files created in this class, no matter what extension
        """
        print('WordEmbedding is running..')
        self.model = None
        self.filename = filename
        self.w2v = w2v
        self.sg = sg
        self.vectordim = vectordim
        self.ngram_len = ngram_len
        self.overlapping = overlapping
        self.windowsize = windowsize
        self.AA_substitute = AA_substitute
        self.sequence_max_len = sequence_max_len
        if model_file:
            self.model_file = True
            self.model = self.load_wv_model(model_file)
        else:
            self.model_file = False
        if emb_matrix_file:
            self.emb_matrix_file = True
            self.emb_matrix = self.load_wv_csv_matrix(emb_matrix_file)
            self.ngramlist = self.create_ngramlist(emb_matrix_file)
        else:
            self.ngramlist = self.create_ngramlist()
            self.emb_matrix_file = False
        self.embedding_vectores = None

    def get_vectores(self):
        x=self.embedding_vectores
        return x

    def get_model(self):
        x=self.model
        return x

    def get_ngramlist(self):
        x = self.ngramlist
        return x

    def get_emb_matrix(self):
        return self.emb_matrix

    def get_rows(self, doc):
        csvreader = csv.reader(doc)
        rows = []
        for row in csvreader:
            rows.append(row)
        return rows


    def aa_substitution(self, sequence):
        """
        Remove all of X amino acids and replace all amino acids B, Z, U and O for N, Q, C and Q.
        """
        sequence.replace('X', '')
        sequence.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
        sequence.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
        sequence.replace('U',
                         'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
        sequence.replace('O', 'K')  # Pyrrolysine to lysine
        return sequence

    def get_list_seqs(self, file, AA_substitute:str = None):
        """
        Get a list of all amino acids sequences from a given file
        :param file: path of the file
        :param AA_substitute: True=Remove all of X amino acids and replace all amino acids B, Z, U and O for N, Q, C and Q.
        :return: list of sequences
        :rtype: list of strings
        """
        #TO-DO meter este
        if AA_substitute != None:
            self.AA_substitute = AA_substitute

        records = self.get_rows(file)
        seqs = []
        #AAList = 'ACDEFGHIKLMNPQRSTVWYUOBXZJ'
        for k in records:
            sequence = k[4]
            if self.AA_substitute == True:
                sequence = self.aa_substitution(sequence)
                seqs.append(sequence)
            else:
                seqs.append(sequence)
        seqs.remove(seqs[0])
        return seqs
####################
    def split_ngrams_non_overlapping(self, seq: str, ngramlen: int = None):
        """
        Split the biological sequence using the non-overlapping method
        :param seq: biological sequence
        :param ngramlen: lenght of the ngram
        :return: list
        """
        #TODO isto só está para ngram = 3
        if ngramlen != None:
            self.ngram_len = ngramlen
        a, b, c = zip(*[iter(seq)] * self.ngram_len), zip(*[iter(seq[1:])] * self.ngram_len), zip(*[iter(seq[2:])] * self.ngram_len)
        str_ngrams = []
        for ngrams in [a, b, c]:
            x = []
            for ngram in ngrams:
                x.append("".join(ngram))
            str_ngrams.append(x)
        return str_ngrams

    def split_ngrams_overlapping(self, seq:list, ngramlen: int = None):
        """
        Split the biological sequence using the overlapping method.
        :param seq: biological sequence
        :param ngramlen: lenght of the ngram
        :return: list of the sequence
        :rtype: list of strings
        """
        # retorna uma lista
        if ngramlen != None:
            self.ngram_len = ngramlen
        chunks = [seq[i:i + self.ngram_len] for i in range(0, len(seq) - self.ngram_len+1, 1)]
        return chunks

    def tokenize_sequences(self, sequences, ngramlen: int = None, overlapping: bool = None):
        """
        Tokenize the biological sequences using the overlapping or non-overlapping methods.
        :param sequences: list of sequences or string of a sequence
        :param ngramlen: lenght of the ngram
        :param overlapping: overlapping method- True: overlapping ; False: non-overlapping
        :return: list of ngrams
        """
        if overlapping != None:
            self.overlapping = overlapping
        if ngramlen != None:
            self.ngram_len = ngramlen

        # print('Tokenizing')
        if self.overlapping == True:
            if type(sequences) == str:
                list_of_ngrams = self.split_ngrams_overlapping(sequences, self.ngram_len)
                return list_of_ngrams
            else:
                seqs_tokenized = []
                for x in sequences:
                    list_of_ngrams = self.split_ngrams_overlapping(x, self.ngram_len)
                    seqs_tokenized.append(list_of_ngrams)
                return seqs_tokenized

        #rv
        elif self.overlapping == False:
            if type(sequences) == str:
                list_of_ngrams = self.split_ngrams_non_overlapping(sequences, self.ngram_len)
                return list_of_ngrams
            else:

                seqs_tokenized = []
                for x in sequences:
                    list_of_ngrams = self.split_ngrams_overlapping(x, self.ngram_len)
                    seqs_tokenized.append(list_of_ngrams)
                return seqs_tokenized


    def sequence_preparation(self,
                             filename: str,
                             list_of_sequences: list,
                             aa_subs: bool = None,
                             overlapping: str=None,
                             ngramlen: int = None):
        """
        Sequence preparation of a given file or list of sequences, tokenizing using different ngrams segmentation methods.
        :param filename: path of the dataset with the sequences
        :param list_of_sequences: list of the biological sequences
        :param aa_subs: True=Remove all of X amino acids and replace all amino acids B, Z, U and O for N, Q, C and Q.
        :param overlapping: overlapping method- True: overlapping ; False: non-overlapping
        :param ngramlen: lenght of the ngram
        :return: list of sequences tokenized
        """
        if aa_subs != None:
            self.AA_substitute = aa_subs
        if overlapping != None:
            self.overlapping = overlapping
        if ngramlen != None:
            self.ngram_len = ngramlen

        if filename is not None:
            file = open(filename)
            list1 = self.get_list_seqs(file, AA_substitute=self.AA_substitute)
        elif list_of_sequences is not None:

            if self.AA_substitute == True:
                list1 = []
                for i in list_of_sequences:
                    list1.append(self.aa_substitution(i))
            else:
                list1 = list_of_sequences
        else:
            print('You should provide a list of sequences!')
            return
        seqs_tokenized = self.tokenize_sequences(list1, ngramlen=self.ngram_len, overlapping=self.overlapping)


        return seqs_tokenized

##################################

    def create_ngramlist(self, matrix = None):
        """
        Create the list of vocabulary, namely the list of all ngrams.
        Obtained through a pre trained matrix or through a pre-created list of all the possible combinations of AA ngrams.
        :param matrix: embedding matrix
        :return: list of all ngrams.
        """
        if matrix:
            x = [i for i in self.emb_matrix.keys()]
        else:
            a_file = open('n_gram_list_only20.json', "r")
            output_dict = json.load(a_file)
            x = output_dict.get(str(self.ngram_len) + '-gram')
            a_file.close()
        return x

    def get_ngramlist_from_newseq(self, sequences: list, ngramlen: int = None, track=True):
        """
        Return a list of all ngrams present in a list of tokenized sequences.
        This method is computationally heavy so it is not advised to use.
        Only use this method instead of the method "create_ngramlist" if you expect a different word
        that is not present in the vocabulary of the pre-created list (with 20 amino acids)
        :param sequences: list of tokenized sequences
        :param ngramlen: length of the ngram
        :param track: if True print the number of sequences scanned.
        :return: set list of ngrams
        """

        print('start geting ngram list')
        if ngramlen != None:
            self.ngram_len = ngramlen
        ngramlist = []
        n = 1
        for sublist in sequences:
            if track is True:
                print('loop:', n, '/', len(sequences))
            else:
                pass
            for x in sublist:
                if x not in ngramlist and len(x) == self.ngram_len:
                    ngramlist.append(x)
            n += 1

        self.ngramlist = ngramlist
        return ngramlist

#######################

    def word2vec_training(self, sequences: list, ngram_list: list =  None, sg:int= None, vectordim: int = None,
                          filename: str = None , windowsize = None , epoch = 10):
        """
        Train, save and set a new word embedding model using word2vec algorithm.
        :param sequences: list of biological sequences.
        :param ngram_list: vocabulary - set list of ngrams.
        :param sg:Training algorithm: 1 for skip-gram; otherwise CBOW.
        :param vector_size:Dimensionality of the word vectors.
        :param filename: Given name to any files created in this class, no matter what extension.
        :param windowsize: Maximum distance between the current and predicted word within a sentence.
        :param epoch:Number of iterations (epochs) over the corpus.
        :return: word embedding vectors.
        """

        print ('Word2Vec..')
        if sg != None:
            self.sg = sg
        if vectordim != None:
            self.vectordim = vectordim
        if ngram_list != None:
            self.ngramlist = ngram_list
        if windowsize != None:
            self.windowsize = windowsize


        callback1 = EpochLogger()
        w2v = Word2Vec(
            sequences,
            window=self.windowsize,
            vector_size=self.vectordim,
            sg=self.sg,
            min_count=1,
            callbacks=[callback1],epochs=epoch)
        word_vectores = w2v.wv
        w2v.wv.save_word2vec_format('word2vec.vector')
        w2v.save(filename)
        print('Model Saved')
        self.model = w2v
        self.embedding_vectores = word_vectores
        return word_vectores

    def load_w2vmodel(self, filename: str):
        """
        Load a pre-trained word2vec model.
        :param filename: Path of the pre-trained model.
        :return: Pre-trained model.
        """
        model = gensim.models.Word2Vec.load(filename)

        self.emb_matrix = KeyedVectors.load(filename, mmap='r')
        self.embedding_vectors = model.wv.vectors
        """if gensim version == 3.8 change use the commented line below"""
        #self.ngramlist = list(model.wv.vocab.keys())
        self.ngramlist = list(model.wv.index_to_key) #gensim == 4.*
        return model

    def fasttext_training(self, sequences: list, ngram_list: list =  None, sg: int = None,
                          filename: str = None , windowsize = None, epoch=10):
        """
        Train, save and set a new word embedding model using fastText algorithm.
        :param sequences: list of biological sequences.
        :param ngram_list: vocabulary - set list of ngrams.
        :param sg:Training algorithm: 1 for skip-gram; otherwise CBOW.
        :param vector_size:Dimensionality of the word vectors.
        :param filename: Given name to any files created in this class, no matter what extension.
        :param windowsize: Maximum distance between the current and predicted word within a sentence.
        :param epoch:Number of iterations (epochs) over the corpus.
        :return: word embedding vectors.
        """
        print('fastText..')
        if sg != None:
            self.sg = sg
        # if vector_size != None:
        #     self.vectordim = vector_size
        if ngram_list != None:
            self.ngramlist = ngram_list
        if windowsize != None:
            self.windowsize = windowsize

        callback1 = EpochLogger()
        ftmodel = FastText(
            sequences,
            window=self.windowsize,
            vector_size=self.vectordim,
            sg=self.sg,
            min_count=1,
            callbacks=[callback1],epochs=epoch)
        word_vectores = ftmodel.wv
        ftmodel.save(filename)
        print ('Model Saved')
        self.model = ftmodel
        self.embedding_vectores = word_vectores
        return  word_vectores

    def load_fasttextmodel(self, filename: str):
        """
        Load a pre-trained fastText model.
        :param filename: Path of the pre-trained model.
        :return: Pre-trained model.
        """
        model = gensim.models.FastText.load(filename)
        self.emb_matrix = KeyedVectors.load(filename, mmap='r')
        self.embedding_vectors = model.wv.vectors
        self.ngramlist = list(model.wv.index_to_key)
        return model
#TODO param**
    def train_wordembedding(self, w2v: int = None, seqs_tokenized: list = None,
                            ngramlist: list = None, sg:int = None ,
                            vectordim = None,
                            ngram_len = None,
                            filename: str = None , save_csv_matrix = True,
                            windowsize = None,epochs = 10):
        """
        Train, save and set a new word embedding model using a specific method.
        :param w2v: Training architecture: 1 for word2vec; otherwise fastText.
        :param seqs_tokenized: Sequences that mus be already tokenized.
        :param ngramlist: vocabulary - set list of ngrams.
        :param sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
        :param filename: Given name to any files created in this method, no matter what extension.
        :param save_csv_matrix: True if you want to save the embedding matrix in a .csv file.
        :param windowsize: Maximum distance between the current and predicted word within a sentence.
        :param epochs: Number of iterations (epochs) over the corpus.
        :return: word embedding vectors.

        """
        print('Start traning word embedding..')

        if sg != None:
            self.sg = sg
        if w2v != None:
            self.w2v = w2v
        if ngramlist != None:
            self.ngramlist = ngramlist
        if windowsize!= None:
            self.windowsize = windowsize
        if vectordim != None:
            self.vectordim = vectordim
        if ngram_len != None:
            self.ngram_len = ngram_len
        if filename:
            self.filename = filename

        path = 'data_processing/'
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)

        if self.w2v == 1:
            if self.filename:
                filename1 ='data_processing/' + str(self.filename) + '.model'
            else:
                if self.sg == 1:
                    filename1 = 'data_processing/word2vecmodel_skipgram.model'
                elif self.sg == 0:
                    filename1 = 'data_processing/word2vecmodel_cbow.model'

            word_vectores = self.word2vec_training(seqs_tokenized,
                                                   sg=self.sg,
                                                   filename=filename1,
                                                   windowsize = windowsize,
                                                   epoch=epochs)

        elif self.w2v == 0:
            if self.filename:
                filename1 = 'data_processing/' + str(self.filename) + '.model'
            else:
                if self.sg == 1:
                    filename1 = 'data_processing/fasttextmodel_skipgram.model'
                elif self.sg == 0:
                    filename1 = 'data_processing/fasttextmodel_cbow.model'

            word_vectores = self.fasttext_training(seqs_tokenized, sg=self.sg,filename= filename1,windowsize = windowsize,epoch=epochs)

        wordvecDict = {i: '' for i in self.ngramlist}

        for gram in self.ngramlist:
            if gram in word_vectores.key_to_index:
                wordvecDict[gram] = np.ndarray.tolist(word_vectores.get_vector(gram))
            else:
                wordvecDict[gram] = [0] * self.vectordim
        if save_csv_matrix == True:
            self.csv_writer_w_pandas_df(wordvecDict, csv_filename=filename)
            return  wordvecDict , word_vectores
        else:
            dataframe1 = pd.DataFrame.from_dict(wordvecDict, orient='index')
            return dataframe1

    def load_wv_model(self,file: str):
        """
        Load a pre-trained word embedding model.
        :param filename: Path of the pre-trained model.
        :return: Pre-trained model.
        """
        #https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
        if self.w2v == 1:
            model = self.load_w2vmodel(file)
            print('Model Loaded')
            self.model = model
            return model
        elif self.w2v == 0:
            model = self.load_fasttextmodel(file)
            print('Model Loaded')
            self.model = model
            return model
        else:
            print('Define word embedding tool word2vec or fastText')



    def load_wv_csv_matrix(self,file):
        """
        Load a pre-trained word embedding model from a .csv file.
        :param filename: Path of the pre-trained model.
        :return: Pre-trained model.
        """
        with open(file) as csvfile:
            spamreader = csv.reader(csvfile,delimiter="\t")
            l = []
            for row in spamreader:
                l.append(row)
            l.pop(0)
            matrix_dict = {}
            for i in l:
                a = str(i.pop(0))
                j = []
                for x in i:
                    j.append(float(x))
                matrix_dict[a] = tuple(j)
            print('--MATRIX LOADED--')

            return matrix_dict


    def save_model_matrix(self, data_dict: dict , csv_filename: str = None):
        """
        Save the embedding matrix in a .csv file.
        :param data_dict: Dictionary of the vectors. keys = ngrams ; values=vectors
        :param csv_filename: Given name to the file created.
        """

        if csv_filename == None:
            if self.w2v == 1:
                if csv_filename == None:
                    if self.sg == 1:
                        csv_filename = 'trained_matrix/word2vecmodel_skipgram_matrix.csv'
                    elif self.sg == 0:
                        csv_filename = 'trained_matrix/word2vecmodel_cbow_matrix.csv'
            elif self.w2v == 0:
                if csv_filename == None:
                    if self.sg == 1:
                        csv_filename = 'trained_matrix/fasttextmodel_skipgram_matrix.csv'
                    elif self.sg == 0:
                        csv_filename = 'trained_matrix/fasttextmodel_cbow_matrix.csv'

        myheaders = ['ngram']
        n = 1
        for i in range(self.vectordim):
            myheaders.append('d' + str(n) )
            n += 1
        with open(csv_filename, 'w', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(myheaders)
            writer.writerows(data_dict.keys())

    def csv_writer_w_pandas_df(self, datadict: dict, csv_filename: str = None):
        """
        Save the embedding matrix in a .csv file using the library pandas.
        :param data_dict: Dictionary of the vectors. keys = ngrams ; values=vectors
        :param csv_filename: Given name to the file created.
        """
        # dict_data: dict with the word vector of every N-gram
        if csv_filename:
            self.filename= csv_filename
        if self.w2v == 1:
            if csv_filename:
                csv_filename = 'data_processing/' + str(self.filename) + '.csv'
            else:
                if self.sg == 1:
                    csv_filename = 'data_processing/word2vecmodel_skipgram_matrix.csv'
                elif self.sg == 0:
                    csv_filename = 'data_processing/word2vecmodel_cbow_matrix.csv'
        elif self.w2v == 0:
            if csv_filename:
                csv_filename = 'data_processing/' + str(self.filename) + '.csv'
            else:
                if self.sg == 1:
                    csv_filename = 'data_processing/fasttextmodel_skipgram_matrix.csv'
                elif self.sg == 0:
                    csv_filename = 'data_processing/fasttextmodel_cbow_matrix.csv'

        dataframe1 = pd.DataFrame.from_dict(datadict, orient='index')
        dataframe1.to_csv(csv_filename, sep='\t')
        print('WordEmbedding Matrix Saved')
####################################
    def get_ngram_vector(self,ngram:str):
        """
        Get ngram vector. Depending on the type of file used (.model or .csv), the way to get the ngram vector
        is different.
        :param ngram: word ngram
        :return: word vector
        """
        if self.model:
            #wordvector = self.emb_matrix.wv.get_vector(ngram, norm=True)
            wordvector = self.emb_matrix.wv.get_vector(ngram)
        elif self.emb_matrix_file:
            wordvector = self.emb_matrix.get(ngram)
        else:
            print('model file not provided!!')
            return
        return wordvector

    def get_sequence_dict_vector(self, sequence):
        """
        Create a dictionary with N-gram as KEY and a N-dim vector as Value.
        :param sequence: biological sequence
        :return: dictionary
        """

        tokenized_sequence = self.tokenize_sequences(sequence)

        dict1 = {}
        for ngram in tokenized_sequence:
            dict1[ngram] = self.get_ngram_vector(ngram)


        return dict1

    def ngram_occurence_counts(self , sequence: str = None):
        """
        Create a dictionary that counts how may times a ngram appears in all sequences given.
        :param sequence: biological sequence
        :return: dictionary
        """
        #faz a contagem dos ngrams
        dict1 = {i: 0 for i in self.ngramlist }
        sequence_toc = self.tokenize_sequences(sequence)

        for i in sequence_toc:
            dict1[i] += 1

        return dict1
##############
    def matrix_method_2(self,sequence = None):
        """
        Create a matrix using the method 2 with shape = (number of sequences, number of words, dimenson of the vector).
        :param sequence: biological sequence
        :return: sequence vector
        """
        dict1 = self.ngram_occurence_counts(sequence)
        matrix = self.emb_matrix

        new_matrix = {}
        for k,v in matrix.items():
            new_matrix[k]=[item * dict1[k] for item in v]

        # my_list = [2, 4, 6]
        # result = [item * 10 for item in my_list]
        # new_matrix = {key: value * int(dict1.get(value)) for key, value in new_matrix.items()}
        return new_matrix

    def vector_method_2(self, sequence = None, ngram_remove_list:list = None):
        """
        Create a vector using the method 2 with shape = (number of sequences , number of words * dimenson of the vector).
        :param sequence: biological sequence
        :return: sequence vector
        """
        matrix = self.matrix_method_2(sequence)
        if ngram_remove_list:
            for k in ngram_remove_list:
                matrix.pop(k, None)


        finalvector = np.array(list(matrix.values())).flatten()

        return finalvector


    def vector_method_3(self, sequence: list = None ):
        """
        Create a vector using the method 3 with shape = (number of sequences , dimenson of the vector).
        :param sequence: biological sequence
        :return: sequence vector
        """

        #multiplica o numero de occurrences do ngram e depois soma
        dict1 = self.ngram_occurence_counts(sequence) #get list de occurrenci de ngrm
        dict2 = self.get_sequence_dict_vector(sequence) #get dict: Key ngram Value Vector da sequencia
        for i in dict2: #multiplica o numero de occurence nos vectores
            multiplier = dict1.get(i)
            listofvaleus = []
            for j in dict2[i]:
                listofvaleus.append(j * int(multiplier))
            dict2[i] = listofvaleus

        dict_values = []
        for i in dict2:
            dict_values.append(dict2.get(i))

        final_vector = [0] * self.vectordim
        for i in dict_values: #soma os vectores todos retornando um vector final de self.vector_dim
            n = 0
            for j in i:
                final_vector[n] += j
                n += 1
        final_vector = np.array(final_vector)
        return final_vector

    def vector_method_1(self, sequence, ngram_remove = False, padding = False):
        """
        Create a vector using the method 1 with shape = (number of sequences , number of words * dimenson of the vector).
        :param sequence: biological sequence
        :param ngram_remove: list of n-grams to remove (can be use when you calculete a varience of the ngram counts)
        :param padding: Padding is an option, when you want to have a fixed input shape.
        :return: sequence vector
        """

        finalvector = []
        tokanized_sequence = self.tokenize_sequences(sequence)

        if ngram_remove:
            for k in ngram_remove:
                if k in ngram_remove:
                    tokanized_sequence.remove(k)
                else: pass

        for ngram in tokanized_sequence:
            finalvector.append(np.array(self.get_ngram_vector(ngram)))

        finalvector = np.array(finalvector)
        if padding == True:
            m = int(self.sequence_max_len) - int(self.ngram_len) + 1
            n = m * int(self.vectordim)
            if len(finalvector) < m:
                dif = m - len(finalvector)
                extravector = np.zeros((dif,self.vectordim))
                finalvector = np.append(finalvector, extravector, axis =0)


        return finalvector

    def convert_seq2vec(self, method:int , sequence: str= None, ngram_remove = None , sequence_max_len = None , padding = False ):
        """
        Convert a biological sequence to a vector through a choosen method of vector creation.
        :param method:
        Method 1:
        Substitute directly the n-grams presented in the sequence by the WE vector. Being K the
        dimension of the word and N the dimension of the WE vector, a sequence of size L will
        be represented by a final vector of (L − k − 1) ∗ N elements. This method preserves the
        spatial information of the location of biological words.
        Method 2:
        In this method, k-mer word frequencies are calculated and multiplied by the corresponding
        WE vectors. A sequence, independent of the size, will be represented by a matrix of
        dimensions Number_of _words ∗ N.
        Method 3:
        All the vectors of Method 2 are summed to reproduce a single vector of dimension N.
        :param sequence: biological sequence
        :param sequence_max_len: Maximum length of the biological sequence
        :param ngram_remove: list of n-grams to remove (can be use when you calculete a varience of the ngram counts)
        :param padding: Padding is an option, when you want to have a fixed input shape.
        :return: sequence vector
        """

        if sequence_max_len:
            self.sequence_max_len = sequence_max_len
            sequence = sequence[0:self.sequence_max_len]
        if method == 1:
            finalvector = self.vector_method_1(sequence, ngram_remove,padding)
            return finalvector

        elif method == 2:
            finalvector = self.vector_method_2(sequence,ngram_remove)
            return finalvector
        elif method == 3:
            finalvector = self.vector_method_3(sequence)
            return finalvector
        else:
            print('define method!')
            return


    def convert_sequences2vec(self, method:int , sequences: list= None, ngram_remove = None , sequence_max_len = None , array_flat= False, padding = False ):
        """
        Convert a biological sequence to a vector through a choosen method of vector creation.
        :param method:
        Method 1:
        Substitute directly the n-grams presented in the sequence by the WE vector. Being K the
        dimension of the word and N the dimension of the WE vector, a sequence of size L will
        be represented by a final vector of (L − k − 1) ∗ N elements. This method preserves the
        spatial information of the location of biological words.
        Method 2:
        In this method, k-mer word frequencies are calculated and multiplied by the corresponding
        WE vectors. A sequence, independent of the size, will be represented by a matrix of
        dimensions Number_of _words ∗ N.
        Method 3:
        All the vectors of Method 2 are summed to reproduce a single vector of dimension N.
        :param sequences:list of biological sequences
        :param array_flat: True if wanted a vector 1 in matrix shape; 3 dimension vector; only used in method 1.
        :param sequence_max_len: Maximum length of the biological sequence
        :param ngram_remove: list of n-grams to remove (can be use when you calculate a variance of the ngram counts)
        :param padding: Padding is an option, when you want to have a fixed input shape.
        :return: sequence vector
        """
        if method == 1:
            vectorized_seqs = []
            for seq in sequences:
                seq_vector = self.convert_seq2vec(method, seq, ngram_remove , sequence_max_len, padding)
                vectorized_seqs.append(seq_vector)

            finalvector = np.array(vectorized_seqs)

            if array_flat == True:
                finalvector = np.reshape(finalvector, (finalvector.shape[0], finalvector.shape[1] * finalvector.shape[2]))
            print('vectors.shape',finalvector.shape)
            return finalvector

        elif method in [2, 3]:
            seqs_vector = []
            for i in sequences:
                vector = self.convert_seq2vec(method=3, sequence=i, ngram_remove=ngram_remove)
                seqs_vector.append(vector)
            seqs_vector = np.array(seqs_vector)
            return seqs_vector
        else:
            print('define method!!')
            return



def main():
    # sequences = ['MLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCAREDPRDAFISTHYASIDLLPTGAVVGTSSMRRQCQLRERRPDLVMRDLRGNIGTRLEKLDKGEYDALILAAAGLKRLDLAHRIRMIIDPTELLPAVGQGVIGIEYRLEDTHILSILAPLHHSATALRVSAERAMNAKLAGGCQVPIGSYAEIEGDQIWLRALVGSPDGSLIIRSEGRAPLSQAEILGQSIANDLLYRGAESILCRAFQVDS',
    #                 'MMRTIKVGSRRSKLAMTQTKWVIQKLKEINPSFAFEIKEIVTKGDRIVDVTLSKVGGKGLFVKEIEQALLNEEIDMAVHSMKDMPAVLPEGLVIGCIPEREDPRDALISKNRVKLSEMKKGAVIGTSSLRRSAQLLIERPDLTIKWIRGNIDTRLQKLETEDYDAIILAAAGLSRMGWKQDVVTEFLEPERCLPAVGQGALAIECRESDEELLALFSQFTDEYTKRTVLAERAFLNAMEGGCQVPIAGYSVLNGQDEIEMTGLVASPDGKIIFKETVTGNDPEEVGKRCAALMADKGAKDLIDRVKRELDED',
    #                 'MNSETLPAELPATLTIASRESRLAMWQAEHVRDALRKLYPACDVKILGMTTRGDQILDRTLSKVGGKGLFVKELESALADGRADLAVHSLKDVPMELPAGFALAAVMSREDPRDAFVSNDYASLDALPAGAVVGTSSLRREAMLRARYPRLDVRPLRGNLDTRLAKLDRGDYAAIILAAAGLKRLGLAARIRALLDVEDSLPAAGQGALGIEIAAGRADVAAWLAPLHDHATALAVEAERAVSRALGGSCEVPLAAHAVWRGDELHLTGSVSTTDGARVLAARAQSRAATAADALALGRAVSDELERQGARAIVDALVAASAQAQKGGA']
    # w2v = WordEmbedding(w2v=0, sg=1)
    # w2v_sequeces = w2v.sequence_preparation(filename='datasets/ecpred_uniprot_uniref_90.csv')
    # w2v.train_wordembedding(seqs_tokenized=w2v_sequeces)
    # w2v = WordEmbedding(w2v=1, sg=1)
    # w2v_sequeces = w2v.sequence_preparation(filename='datasets/ecpred_uniprot_uniref_90.csv')
    # w2v.train_wordembedding(seqs_tokenized=w2v_sequeces)
    # w2v = WordEmbedding(w2v=0, sg=0)
    # w2v_sequeces = w2v.sequence_preparation(filename='datasets/ecpred_uniprot_uniref_90.csv')
    # w2v.train_wordembedding(seqs_tokenized=w2v_sequeces)
    w2v = WordEmbedding(w2v=1, sg=1 , vectordim=100 , ngram_len= 1 ,windowsize=5)
    w2v_sequeces = w2v.sequence_preparation(filename='/home/igomes/Bumblebee/datasets/ecpred_uniprot_uniref_90_no_negatives.csv' )
    w2v.train_wordembedding(seqs_tokenized=w2v_sequeces , filename = 'w2v_sg_1gram_100dim_10ep_nneg' , epochs= 10)
    w2v = WordEmbedding(w2v=1, sg=1, vectordim=100, ngram_len=2, windowsize=5)
    w2v_sequeces = w2v.sequence_preparation(
        filename='/home/igomes/Bumblebee/datasets/ecpred_uniprot_uniref_90_no_negatives.csv')
    w2v.train_wordembedding(seqs_tokenized=w2v_sequeces, filename='w2v_sg_2gram_100dim_10ep_nneg', epochs=10)


def main2():
    sequence = 'MLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCIMLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCI'
    sequences = ['MLNNILKIATRQSPLAIWQANYVRNQLLSFYPTLLIELVPIVTSGDLILDKPLMKAGGKRLFIKELEQAMLERRADIAVHSMKDITISFPEGIGLAVLCAREDPRDAFISTHYASIDLLPTGAVVGTSSMRRQCQLRERRPDLVMRDLRGNIGTRLEKLDKGEYDALILAAAGLKRLDLAHRIRMIIDPTELLPAVGQGVIGIEYRLEDTHILSILAPLHHSATALRVSAERAMNAKLAGGCQVPIGSYAEIEGDQIWLRALVGSPDGSLIIRSEGRAPLSQAEILGQSIANDLLYRGAESILCRAFQVDS',
                     'MMRTIKVGSRRSKLAMTQTKWVIQKLKEINPSFAFEIKEIVTKGDRIVDVTLSKVGGKGLFVKEIEQALLNEEIDMAVHSMKDMPAVLPEGLVIGCIPEREDPRDALISKNRVKLSEMKKGAVIGTSSLRRSAQLLIERPDLTIKWIRGNIDTRLQKLETEDYDAIILAAAGLSRMGWKQDVVTEFLEPERCLPAVGQGALAIECRESDEELLALFSQFTDEYTKRTVLAERAFLNAMEGGCQVPIAGYSVLNGQDEIEMTGLVASPDGKIIFKETVTGNDPEEVGKRCAALMADKGAKDLIDRVKRELDED',
                    'MNSETLPAELPATLTIASRESRLAMWQAEHVRDALRKLYPACDVKILGMTTRGDQILDRTLSKVGGKGLFVKELESALADGRADLAVHSLKDVPMELPAGFALAAVMSREDPRDAFVSNDYASLDALPAGAVVGTSSLRREAMLRARYPRLDVRPLRGNLDTRLAKLDRGDYAAIILAAAGLKRLGLAARIRALLDVEDSLPAAGQGALGIEIAAGRADVAAWLAPLHDHATALAVEAERAVSRALGGSCEVPLAAHAVWRGDELHLTGSVSTTDGARVLAARAQSRAATAADALALGRAVSDELERQGARAIVDALVAASAQAQKGGA']
    #--VARIANCE
    # data1 = pd.read_csv('counts_por_seq.csv')
    # data1 = data1.drop(data1.columns[[0, 1]], axis=1)
    # print('Start Var')
    # sel = VarianceThreshold(0.3)
    # transf = sel.fit_transform(data1)
    # # original dataset without columns
    # ngram_selected = []
    # column_selected = sel.get_support(indices=True)
    # for i in column_selected:
    #     ngram_selected.append(data1.columns[i])
    #print(ngram_selected)
    #WV
    w2vmodel = WordEmbedding(emb_matrix_file='protVec_100d_3grams.csv')
    vector = w2vmodel.convert_seq2vec(method=1, sequence=sequence) #,ngram_remove= ngram_selected)
    print(vector)
    #
    # l = []
    # for seq in sequences:
    #     vector = w2vmodel.convert_seq2vec(method=1, sequence= seq, matrix= True)
    #     l.append(vector)
    #     #print(vector)
    # l = np.array(l)
    #print(l.shape)

def main3():
    wv = WordEmbedding(w2v = 0)
    model = wv.load_fasttextmodel('/home/igomes/Bumblebee/data_processing/fasttest_skip_3gram_100dim_10ep_nneg.model')
    print(model)
    #measure of similarity analogies
    x = model.wv.most_similar('MII')
    print(x)
def create_json_all_AA():
    AAList = 'ACDEFGHIKLMNPQRSTVWY'
    csv_columns = ['1-gram','2-gram','3-gram','4-gram','5-gram']
    dict1 = {'1-gram': [] , '2-gram': [] , '3-gram': [] , '4-gram' : [] , '5-gram': []}
    for aa in AAList:
        dict1['1-gram'].append(aa)
        for aab in AAList:
            dict1['2-gram'].append(aa+aab)
            for aaab in AAList:
                dict1['3-gram'].append(aa + aab + aaab)
                for aaaab in AAList:
                    dict1['4-gram'].append(aa + aab + aaab + aaaab)
                    for aaaaab in AAList:
                        dict1['5-gram'].append(aa + aab + aaab + aaaab + aaaaab)

    # with open('list_of_ngrams1.csv', 'w') as f:
    #     for key in dict1.keys():
    #         f.write("%s,%s\n" % (key, dict1[key]))
    a_file = open("n_gram_list_only20.json", "w")
    json.dump(dict1, a_file)
    a_file.close()
    print (dict1)

if __name__ == "__main__":
    #create_json_all_AA()
    main()


