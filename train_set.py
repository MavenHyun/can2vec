import numpy as np
import pandas as pd
import statistics as st

class data_set:
    def __init__(self, cancer_type):
        self.type = cancer_type

    def data_extract(self):
        df = pd.read_csv(self.type + "_features.tsv", '\t')
        df.fillna(0, inplace=True)
        c, m, v, r = 0, 0, 0, 0

        self.A = np.array(df.values[:, :]).transpose()

        for col in self.A[0, :]:
            if "_Clinical" in col:
                c += 1
            elif "_Mut" in col:
                m += 1
            elif "_CNV" in col:
                v += 1
            else:
                r += 1

        self.X = {'all': np.array(df.values[:c + m + v + r, 1:]).transpose(),
                  'cli': np.array(df.values[:c, 1:]).transpose(),
                  'mut': np.array(df.values[c:c + m, 1:]).transpose(),
                  'CNV': np.array(df.values[c + m:c + m + v, 1:]).transpose(),
                  'mRNA1': np.array(df.values[c + m + v:c + m + v + 4000, 1:]).transpose(),
                  'mRNA2': np.array(df.values[c + m + v + 4000:c + m + v + 8000, 1:]).transpose(),
                  'mRNA3': np.array(df.values[c + m + v + 8000:c + m + v + 12000, 1:]).transpose(),
                  'mRNA4': np.array(df.values[c + m + v + 12000:c + m + v + r, 1:]).transpose(),
                  'mRNA': np.array(df.values[c + m + v:c + m + v + r, 1:]).transpose()}

        df = pd.read_csv(self.type + "_survival.tsv", '\t')
        raw_input = df.values[:, 1:]
        self.Y = np.array(raw_input).transpose()

        df = pd.read_csv(self.type + "_censored.tsv", '\t')
        raw_input = df.values[:, 1:]
        self.Z = np.array(raw_input).transpose()

    def data_preprocess(self):

        for i in range(self.X['mRNA1'].shape[1]):
            min, max = np.min(self.X['mRNA1'][:, i]), np.max(self.X['mRNA1'][:, i])
            if max - min != 0:
                for j in range(self.X['mRNA1'].shape[0]):
                    self.X['mRNA1'][j, i] = (self.X['mRNA1'][j, i] - min) / (max - min)
        
        for i in range(self.X['mRNA2'].shape[1]):
            min, max = np.min(self.X['mRNA2'][:, i]), np.max(self.X['mRNA2'][:, i])
            if max - min != 0:
                for j in range(self.X['mRNA2'].shape[0]):
                    self.X['mRNA2'][j, i] = (self.X['mRNA2'][j, i] - min) / (max - min)
                    
        for i in range(self.X['mRNA3'].shape[1]):
            min, max = np.min(self.X['mRNA3'][:, i]), np.max(self.X['mRNA3'][:, i])
            if max - min != 0:
                for j in range(self.X['mRNA3'].shape[0]):
                    self.X['mRNA3'][j, i] = (self.X['mRNA3'][j, i] - min) / (max - min)
                    
        for i in range(self.X['mRNA4'].shape[1]):
            min, max = np.min(self.X['mRNA4'][:, i]), np.max(self.X['mRNA4'][:, i])
            if max - min != 0:
                for j in range(self.X['mRNA4'].shape[0]):
                    self.X['mRNA4'][j, i] = (self.X['mRNA4'][j, i] - min) / (max - min)
        
        for i in range(self.X['mRNA'].shape[1]):
            min, max = np.min(self.X['mRNA'][:, i]), np.max(self.X['mRNA'][:, i])
            if max - min != 0:
                for j in range(self.X['mRNA'].shape[0]):
                    self.X['mRNA'][j, i] = (self.X['mRNA'][j, i] - min) / (max - min)

        '''
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])
        for i in range(self.Y.shape[0]):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv
        '''
        


