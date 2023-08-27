from .word_embedding import WordEmbedding as wv
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from random import *
_inst = Random()
seed = _inst.seed
random = _inst.random
uniform = _inst.uniform
triangular = _inst.triangular
randint = _inst.randint

NGRAM_PROPERTIES = {
    'A': [71.0788  , 67. , 67. , 7.   , 47. , 6.01 ],
    'C': [103.1448 , 86. , 86. , 8.   , 52. , 5.05 ],
    'D': [115.0886 , 91. , 91. , 18.  , -18., 2.85 ],
    'E': [129.1155 , 109., 109., 17.  , -31., 3.15 ],
    'F': [147.1766 , 135., 135., 4.   , 100., 5.49 ],
    'G': [57.052   , 48. , 48. , 9.   , 0.  , 6.06 ],
    'H': [137.1412 , 118., 118., 13.  , -42., 7.6  ],
    'I': [113.1595 , 124., 124., 2.   , 99. , 6.05 ],
    'K': [128.1742 , 135., 135., 15.  , -23., 9.6  ],
    'L': [113.1595 , 124., 124., 1.   , 100., 6.01 ],
    'M': [131.1986 , 124., 124., 5.   , 74. , 5.74 ],
    'N': [114.1039 , 96. , 96. , 16.  , -41., 5.41 ],
    'P': [97.1167  , 90. , 90. , 11.5 , -46., 6.3  ],
    'Q': [128.1308 , 114., 114., 14.  , 8.  , 5.65 ],
    'R': [156.1876 , 148., 148., 19.  , 41. , 10.76],
    'S': [87.0782  , 73. , 73. , 12.  ,-7.  , 5.68 ],
    'T': [101.1051 , 93. , 93. , 11.  , 13. , 5.6  ],
    'V': [99.1326  , 105., 105., 3.   , 79. , 6.   ],
    'W': [186.2133 , 163., 163., 6.   , 97. , 5.89 ],
    'Y': [163.176  , 141., 141., 10.  , 63. , 5.64 ],
    'X': [142.67295, 134., 134., 4.5  , 88. , 5.45 ],
    'U': [168.064, 0., 0., 0  , 0. , 0. ],
    'O': [255.313, 0., 0., 0  , 0. , 0. ]
}
class ModelTsne:
    """
    ModelTsne train, load and create plots for visualization of the embedding vectors.
    """
    def __init__(self):
        """
        Construct a new 'ModelTsne' object.
        """
        print ('TSNE is running..')

    def make_tsne(self, tokens,
                  filename: str = 'tsne_model.sav',
                  perplexity=500,
                  n_components=2,
                  init='pca',
                  n_iter=1000,
                  random_state=23,
                  verbose=1,
                  learning_rate=1000):
        """
        Trains, save and return a t-SNE model.
        :param tokens: input tokens
        :param filename: name given to the file
        :param perplexity: The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significantly different results. The perplexity must be less that the number of samples.
        :param n_components: Dimension of the embedded space.
        :param init: initialization of embedding.
        :param n_iter: Maximum number of iterations for the optimization
        :param random_state: Determines the random number generator.
        :param verbose: Verbosity level.
        :param learning_rate:
        :return: t-SNE trained model.
        :rtype: pickle
        """

        tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter, random_state=random_state, verbose=verbose,
                          learning_rate=learning_rate)
        tsne_fit = tsne_model.fit_transform(tokens)

        with open(filename, 'wb') as files:
            pickle.dump(tsne_fit, files)

        return tsne_fit

    def load_tsne(self, filename:str ='trained_models/tsne_model.sav'):
        """
        Load a t-SNE model.
        :param filename: path to file
        :return: pre-trained model.
        :rtype: pickle
        """
        with open(filename, 'rb') as f:
            lr = pickle.load(f)
        print('Model Loaded')
        return lr

    def calculate_property(self, label):
        """
        Calculates the biological properties for a given amino acid.
        properties: Mass, Volume, Van der Waals Volume, Polarity, Hydrophobicity, Charge
        :param label: given amino-acid.
        :return: list of properties values.
        """
        label2 = []
        for aa in label:

            split_to_char = list(aa)
            sum_properties = np.array([0., 0., 0., 0., 0., 0.])
            for char in split_to_char:
                sum_properties += np.array(self.pick_key(char))
            sum_properties /= 3.
            sum_properties = list(sum_properties)
            label2.append(sum_properties)

        return label2

    def pick_key(self, char):
        """
        Replaces the amino acids B,Z,J with a random amino acid present in the rand_dict dictionary.
        :param char: target amino acid
        :return: new amino acid
        :rtype: str
        """
        rand_dict = {1: 'N', 2: 'D', 3: 'E', 4: 'Q', 5: 'L', 6: 'I'}
        try:
            return NGRAM_PROPERTIES[char]
        # return NGRAM_PROPERTIES[char]
        except:
            if char == 'B':
                return NGRAM_PROPERTIES[rand_dict[randint(1, 2)]]
            elif char == 'Z':
                return NGRAM_PROPERTIES[rand_dict[randint(3, 4)]]
            elif char == 'J':
                return NGRAM_PROPERTIES[rand_dict[randint(5, 6)]]

    def visualization(self, X_tsne, label,  filename):
        """
        Visualization of the 6 graphics from the
        attributes calculated for tripeptides as protvec
        :param X_tsne:
        :param label:
        :param filename:
        :return:
        """
        # load final_embedding data
        print('Visualization')
        fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
        # set marker size
        marker_size = 1

        # set scatter
        g1 = axarr[0, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, label[:, 0])
        axarr[0, 0].set_title("Mass")
        fig.colorbar(g1, ax=axarr[0, 0])

        g2 = axarr[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, label[:, 1])
        axarr[0, 1].set_title("Volume")
        fig.colorbar(g2, ax=axarr[0, 1])

        g3 = axarr[0, 2].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, label[:, 2])
        axarr[0, 2].set_title("Van der Waals Volume")
        fig.colorbar(g3, ax=axarr[0, 2])

        g4 = axarr[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, label[:, 3])
        axarr[1, 0].set_title("Polarity")
        fig.colorbar(g4, ax=axarr[1, 0])

        g5 = axarr[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, label[:, 4])
        axarr[1, 1].set_title("Hydrophobicity")
        fig.colorbar(g5, ax=axarr[1, 1])

        g6 = axarr[1, 2].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, label[:, 5])
        axarr[1, 2].set_title("Charge")
        fig.colorbar(g6, ax=axarr[1, 2])


        plt.savefig(filename)
        plt.show()


    def tsne_plot_2D(self, tsne_model= None, model= None, labels= None , filename: str = 'embedding_2d.jpg'):
        """
        Visualization of embedding vectors from a t-SNE model.
        :param tsne_model: pre trained TSNE model
        :param model: input a model to train a TSNE model w/ them
        :param labels: amino acids corresponding to the vectors
        :param filename: name given to the file
        :return: 2D-plot of the TSNE model.
        """

        if tsne_model is None:
            tsne_fit , labels = self.make_tsne(model)
        else:
            tsne_fit = self.load_tsne(tsne_model)

        x = []
        y = []
        for value in tsne_fit:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))

        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        # colors = cm.rainbow(np.linspace(0, 1, 1))
        # plt.scatter(x, y, c=colors, alpha=0.1, label='Coisa')
        # for i, word in enumerate(tsne_fit):
        #     plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
        #                  textcoords='offset points', ha='right', va='bottom', size=10)
        plt.legend(loc=4)
        plt.grid(True)
        plt.savefig(filename)
        plt.show()

    def tsne_plot_3D(self, tsne_model= None,
                     model= None,
                     title:str = 'Visualizing Embeddings using t-SNE',
                     label:str = 'Dataset: unknown',
                     filename: str = 'embbed_img_3d.jpg' , a:int = 0.1):
        if tsne_model == None:
            tsne_fit, labels = self.make_tsne(model)
        else:
            tsne_fit = self.load_tsne(tsne_model)

        fig = plt.figure()
        colors = cm.rainbow(np.linspace(0, 1, 1))
        plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], tsne_fit[:, 2], c=colors, alpha=a,
                    label=label)
        plt.legend(loc=4)
        plt.title(title)
        plt.savefig(filename)
        plt.show()

    def tsne_plot(self, tsne_fit= None, labels= None , filename: str = 'embbed_img_2d.jpg'):
        ''' input:
        tsne_model: pre trained TSNE model
        model: input a model to train a TSNE model w/ them
        '''

        # if tsne_model == None:
        #     tsne_fit , labels = self.make_tsne(model)
        # else:
        #     tsne_fit = self.load_tsne(tsne_model)

        x = []
        y = []
        for value in tsne_fit:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))

        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)
        plt.show()


    def visualization_2(self, X_tsne , y , filename):
        # load final_embedding data
        print('Visualization')
        fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
        # set marker size
        marker_size = 1

        # set scatter
        g1 = axarr[0, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], marker_size, y)
        axarr[0, 0].set_title("A")
        fig.colorbar(g1)

        plt.savefig(filename)
        plt.show()

    def visualization_3(self, X_1 , y1, X_2 , y2,X_3 , y3,X_4 , y4,X_5 , y5,X_6 , y6 , filename):
        # load final_embedding data
        print('Visualization')
        fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
        # set marker size
        marker_size = 1

        # set scatter
        g1 = axarr[0, 0].scatter(X_1[:, 0], X_1[:, 1], marker_size, y1)
        axarr[0, 0].set_title("w2v Skip-Gram 3gram 100d")
        fig.colorbar(g1, ax=axarr[0, 0])

        g2 = axarr[0, 1].scatter(X_2[:, 0], X_2[:, 1], marker_size, y2)
        axarr[0, 1].set_title("w2v CBow 3gram 100d")
        fig.colorbar(g2, ax=axarr[0, 1])

        g3 = axarr[0, 2].scatter(X_3[:, 0], X_3[:, 1], marker_size, y3)
        axarr[0, 2].set_title("FastText Ski-pgram 3gram 100d")
        fig.colorbar(g3, ax=axarr[0, 2])

        g4 = axarr[1, 0].scatter(X_4[:, 0], X_4[:, 1], marker_size, y4)
        axarr[1, 0].set_title("w2v Skip-Gram 3gram 50d")
        fig.colorbar(g4, ax=axarr[1, 0])

        g5 = axarr[1, 1].scatter(X_5[:, 0], X_5[:, 1], marker_size, y5)
        axarr[1, 1].set_title("w2v Skip-Gram 3gram 20d")
        fig.colorbar(g5, ax=axarr[1, 1])

        g6 = axarr[1, 2].scatter(X_6[:, 0], X_6[:, 1], marker_size, y6)
        axarr[1, 2].set_title("w2v SkipGram with negatives")
        fig.colorbar(g6, ax=axarr[1, 2])
        plt.savefig(filename)
        plt.show()




def main():
    w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_skip_3gram_100dim_10ep_nneg.csv')
    words = w2v.get_emb_matrix()
    result = words.values()
    data = list(result)
    numpyArray = np.array(data)
    result = words.keys()
    data = list(result)
    y = np.array(data)
    tsne = ModelTsne()
    y = tsne.calculate_property(y)
    print(y)
    #tsne.tsne_plot_2D(model='/home/igomes/Bumblebee/trained_models/tsne_model.sav',labels=numpyArray2,filename='embedding_protVec_2d')
    model = tsne.make_tsne(tokens=numpyArray)
    tsne.tsne_plot(tsne_fit=model,labels=y,filename='embedding_bumble_2d')

def main2():
    #w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/protVec_100d_3grams.csv')
    w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_skip_3gram_100dim_10ep_nneg.csv')
    words = w2v.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())

    numpyArray = np.array(vectors)
    # tsne.tsne_plot_2D(model='/home/igomes/Bumblebee/trained_models/tsne_model.sav',labels=numpyArray2,filename='embedding_protVec_2d')
    model = tsne.make_tsne(tokens=numpyArray)

    ngrams = words.keys()
    y = tsne.calculate_property(ngrams)
    y = np.array(y)
    # set scatter

    tsne.visualization(X_tsne=model, label = y,filename='embedding_visualization_bumble')
def main3():
    #w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/protVec_100d_3grams.csv')
    w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_skip_3gram_100dim_10ep_nneg.csv')
    words = w2v.get_emb_matrix()
    words.pop('<unk>', None)
    #words.pop('WWW', None)
    tsne = ModelTsne()


    vectors = list(words.values())
    numpyArray = np.array(vectors)
    #tsne.tsne_plot_2D(model='/home/igomes/Bumblebee/trained_models/tsne_model.sav',labels=numpyArray2,filename='embedding_protVec_2d')
    model = tsne.make_tsne(tokens=numpyArray)

    ngrams = words.keys()
    df = pd.read_csv('tripeptid.csv', index_col=[0], names=['ngram', 'value'])
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    df2 = ngrams.merge(df, how='inner', on='ngram')
    list1 = list(df2['value'])

    tsne.visualization_2(X_tsne=model, y = list1 ,filename='embedding_visualization_docking scores')

def main4():
    df_tri = pd.read_csv('tripeptid.csv', index_col=[0], names=['ngram', 'value'])
    w2v1 = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_skip_3gram_100dim_10ep_nneg.csv')
    w2v2 = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_cbow_3gram_100dim_10ep_nneg.csv')
    w2v3 = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/fasttest_skip_3gram_100dim_10ep_nneg.csv')
    w2v4 = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_sg_3gram_50dim_10ep_nneg.csv')
    w2v5 = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_sg_3gram_20dim_10ep_nneg.csv')
    w2v6 = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_skip_3gram_100dim_10ep.csv')
    words = w2v1.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())
    numpyArray = np.array(vectors)
    model1 = tsne.make_tsne(tokens=numpyArray)
    # print(model1)
    # print(model1.shape)
    ngrams = words.keys()
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    # print(ngrams)
    df1 = ngrams.merge(df_tri, how='inner', on='ngram')
    list1 = list(df1['value'])
    # print(df1)
    # print(list1)


    #2

    words = w2v2.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())
    numpyArray = np.array(vectors)
    model2 = tsne.make_tsne(tokens=numpyArray)
    ngrams = words.keys()
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    df2 = ngrams.merge(df_tri, how='inner', on='ngram')
    list2 = list(df2['value'])
    #3
    words = w2v3.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())
    numpyArray = np.array(vectors)
    model3 = tsne.make_tsne(tokens=numpyArray)
    ngrams = words.keys()
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    df3 = ngrams.merge(df_tri, how='inner', on='ngram')
    list3 = list(df3['value'])
    #4
    words = w2v4.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())
    numpyArray = np.array(vectors)
    model4 = tsne.make_tsne(tokens=numpyArray)
    ngrams = words.keys()
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    df4 = ngrams.merge(df_tri, how='inner', on='ngram')
    list4 = list(df4['value'])
    #5
    words = w2v5.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())
    numpyArray = np.array(vectors)
    model5 = tsne.make_tsne(tokens=numpyArray)
    ngrams = words.keys()
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    df5 = ngrams.merge(df_tri, how='inner', on='ngram')
    list5 = list(df5['value'])
    #6
    words = w2v6.get_emb_matrix()
    words.pop('<unk>', None)
    tsne = ModelTsne()
    vectors = list(words.values())
    numpyArray = np.array(vectors)
    model6 = tsne.make_tsne(tokens=numpyArray)
    ngrams = words.keys()
    ngrams = pd.DataFrame(ngrams, columns=['ngram'])
    df6 = ngrams.merge(df_tri, how='inner', on='ngram')
    list6 = list(df6['value'])

    tsne.visualization_3(X_1=model1, y1 = list1
                        ,X_2=model2, y2 = list2,
                         X_3=model3, y3 = list3,
                         X_4=model4, y4 = list4,
                         X_5=model5, y5 = list5,
                         X_6=model6, y6 = list6,
                         filename='embedding_visualization_docking_scores')

if __name__ == "__main__":
    main()

