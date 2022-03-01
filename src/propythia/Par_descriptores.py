"""
##############################################################################

A class  used for computing different types of protein descriptors parallelized.
It contains descriptors from packages pydpi, biopython, pfeature and modlamp.

Authors: Miguel Barros

Date: 02/2022

Email:

##############################################################################
"""
import pandas as pd
from joblib import Parallel, delayed
from propythia.descriptors import Descriptor

class Par_descritores():
    def __init__(self, dataset, col: str, merge : bool = False ):
        """
        Constructor

        :param dataset: Pandas dataframe
        :param col: column in the dataframe which contains the protein sequence
        """
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else: raise Exception('Parameter dataframe must be a pandas dataframe')
        self.col = col
        self.merge = merge

    def run(self,list_of_functions=None, ph: float = 7, amide: bool = False,lamda_paac: int = 10,
            weight_paac: float = 0.05,lamda_apaac: int = 10, weight_apaac: float = 0.05, AAP=None,
            maxlag_socn: int = 45, maxlag_qso: int = 30, weight_qso: float = 0.1,distancematrix=None,
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

        r = Parallel(n_jobs=n_jobs)(delayed(self.par_adaptable)(
                seq,list_of_functions,ph, amide,lamda_paac, weight_paac,lamda_apaac, weight_apaac, AAP, maxlag_socn,
                maxlag_qso, weight_qso,distancematrix, window, scalename, scalename_arc,angle, modality, prof_type, tricomp)
                for seq in self.dataset[self.col])
        self.feature = pd.DataFrame(r)
        if self.merge:
            self.feature = self.dataset.merge(self.feature, how='left', on='sequence')
        return self.feature

    def par_adaptable(self,seq,list_of_functions,ph, amide,lamda_paac, weight_paac,lamda_apaac, weight_apaac, AAP,
                      maxlag_socn,maxlag_qso, weight_qso,distancematrix, window, scalename, scalename_arc,angle,
                      modality, prof_type, tricomp):
        """
        Function to calculate user selected descriptors for each sequence in the dataset, obtain by the class descriptor.
        all parameters are received from the run function.

        !!FUNCTION NOT INTENDED TO BE USE DIRECTLY!!

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
        :return: dictionary for the sequence with all descriptors.
        """
        protein = Descriptor(seq)
        feature = protein.adaptable(list_of_functions,ph, amide,lamda_paac, weight_paac,lamda_apaac, weight_apaac, AAP,
                                    maxlag_socn,maxlag_qso, weight_qso,distancematrix, window, scalename, scalename_arc,
                                    angle, modality, prof_type, tricomp)
        res = {'sequence': seq}
        res.update(feature)
        return res

    def toDataframe(self,name : str):
        """
        Function to save the dataframe into a csv file

        :param name: name for the file (file extension not needed)

        :return: creates a csv file
        """
        df = pd.DataFrame(self.feature)
        df.to_csv(name + '.csv')

