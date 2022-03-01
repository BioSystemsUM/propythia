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
from multiprocessing import cpu_count
from propythia.descriptors import Descriptor

class Par_descritores():
    def __init__(self, dataset, col):
        self.dataset = dataset
        self.col = col

    def run(self, list_of_functions=None, ph: float = 7, amide: bool = False,
                  lamda_paac: int = 10, weight_paac: float = 0.05,
                  lamda_apaac: int = 10, weight_apaac: float = 0.05, AAP=None, maxlag_socn: int = 45,
                  maxlag_qso: int = 30, weight_qso: float = 0.1,
                  distancematrix=None, window: int = 7, scalename: str = 'Eisenberg', scalename_arc: str = 'peparc',
                  angle: int = 100, modality: str = 'max', prof_type: str = 'uH', tricomp: bool = False):

        self.feature = Parallel(n_jobs=int(0.8 * cpu_count()))(delayed(self.par_adaptable)(
                seq,list_of_functions,ph, amide,lamda_paac, weight_paac,lamda_apaac, weight_apaac, AAP, maxlag_socn,
                maxlag_qso, weight_qso,distancematrix, window, scalename, scalename_arc,angle, modality, prof_type, tricomp)
                for seq in self.dataset[self.col])
        return self.feature

    def par_adaptable(self,seq,list_of_functions,ph, amide,lamda_paac, weight_paac,lamda_apaac, weight_apaac, AAP,
                      maxlag_socn,maxlag_qso, weight_qso,distancematrix, window, scalename, scalename_arc,angle,
                      modality, prof_type, tricomp):

        protein = Descriptor(seq)
        feature = protein.adaptable(list_of_functions,ph, amide,lamda_paac, weight_paac,lamda_apaac, weight_apaac, AAP,
                                    maxlag_socn,maxlag_qso, weight_qso,distancematrix, window, scalename, scalename_arc,
                                    angle, modality, prof_type, tricomp)
        res = {'sequence': seq}
        res.update(feature)
        return res

    def toDataframe(self,name : str):
        df = pd.DataFrame(self.feature)
        df.to_csv(name)

