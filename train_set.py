import numpy as np
import pandas as pd
import statistics as st

class train_set:
    def __init__(self, fn, fn2):
        self.file_name, self.file_name2 = fn, fn2

    def data_preprocess(self):
        df = pd.read_csv(self.file_name, '\t')
        df.fillna(0, inplace=True)

        self.X, self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA = np.array(df.values[:18872, 1:]).transpose(), \
                                                                    np.array(df.values[:19, 1:]).transpose(), \
                                                                    np.array(df.values[19:128, 1:]).transpose(), \
                                                                    np.array(df.values[128:2348, 1:]).transpose(), \
                                                                    np.array(df.values[2348:18872, 1:]).transpose()
        mean, stdv = st.mean(self.X_categ[:, 0]), st.stdev(self.X_categ[:, 0])
        for i in range(self.X_categ.shape[0]):
            self.X_categ[i, 0] = (self.X_categ[i, 0] - mean) / stdv
        for i in range(self.X_mRNA.shape[1]):
            mean, stdv = st.mean(self.X_mRNA[:, i]), st.stdev(self.X_mRNA[:, i])
            if (stdv != 0):
                for j in range(self.X_mRNA.shape[0]):
                    self.X_mRNA[j, i] = (self.X_mRNA[j, i] - mean) / stdv

        df = pd.read_csv(self.file_name2, '\t')
        raw_input = df.values[:, 1:]
        self.Y = np.array(raw_input).transpose()
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])
        for i in range(self.Y.shape[0]):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv
