import numpy as np
import pandas as pd
import statistics as st

class data_set:
    def __init__(self, f, s, n):
        self.file_name, self.file_name2 , self.num_train =  f, s, n

    def data_extract(self):
        df = pd.read_csv(self.file_name, '\t')
        df.fillna(0, inplace=True)
        c, m, v, r = 0, 0, 0, 0

        self.Z = np.array(df.values[:, :]).transpose()
        for col in self.Z[0, :]:
            if "_Clinical" in col:
                c += 1
            elif "_Mut" in col:
                m += 1
            elif "_CNV" in col:
                v += 1
            else:
                r += 1

        self.X = np.array(df.values[:c + m + v + r, 1:]).transpose()
        self.X_cli = np.array(df.values[:c, 1:]).transpose()
        self.X_mut = np.array(df.values[c:c + m, 1:]).transpose()
        self.X_CNV = np.array(df.values[c + m:c + m + v, 1:]).transpose()
        self.X_mRNA = np.array(df.values[c + m + v:c + m + v + r, 1:]).transpose()

    def data_preprocess(self):
        mean, stdv = st.mean(self.X_cli[:, 0]), st.stdev(self.X_cli[:, 0])
        for i in range(self.X_cli.shape[0]):
            self.X_cli[i, 0] = (self.X_cli[i, 0] - mean) / stdv
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

    def data_split(self):
        [self.X, self.XX] = np.split(self.X, [self.num_train], axis=0)
        [self.X_cli, self.XX_cli] = np.split(self.X_cli, [self.num_train], axis=0)
        [self.X_mut, self.XX_mut] = np.split(self.X_mut, [self.num_train], axis=0)
        [self.X_CNV, self.XX_CNV] = np.split(self.X_CNV, [self.num_train], axis=0)
        [self.X_mRNA, self.XX_mRNA] = np.split(self.X_mRNA, [self.num_train], axis=0)
