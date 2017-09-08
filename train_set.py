import numpy as np
import pandas as pd
import statistics as st

class data_set:
    def __init__(self, cancer_type, num_train, num_valid):
        self.type = cancer_type
        self.train = num_train
        self.valid = num_valid + num_train
        self.samples, self.features = 0, 0
        self.c, self.m, self.v, self.r = 0, 0, 0, 0

    def data_extract(self):
        df = pd.read_csv(self.type + "_features.tsv", '\t')
        df.fillna(0, inplace=True)

        self.A = np.array(df.values[:, :]).transpose()

        for col in self.A[0, :]:
            if "_Clinical" in col:
                self.c += 1
            elif "_Mut" in col:
                self.m += 1
            elif "_CNV" in col:
                self.v += 1
            else:
                self.r += 1

        self.X = {'all': np.array(df.values[:self.c + self.m + self.v + self.r, 1:]),
                  'cli': np.array(df.values[:self.c, 1:]),
                  'mut': np.array(df.values[self.c:self.c + self.m, 1:]),
                  'CNV': np.array(df.values[self.c + self.m:self.c + self.m + self.v, 1:]),
                  'mRNA': np.array(df.values[self.c + self.m + self.v:self.c + self.m + self.v + self.r, 1:])}
        
        self.F = {'all': self.X['all'].shape[0],
                  'cli': self.X['cli'].shape[0],
                  'mut': self.X['mut'].shape[0],
                  'CNV': self.X['CNV'].shape[0],
                  'mRNA': self.X['mRNA'].shape[0] }
        self.samples = self.X['all'].shape[1]

        df = pd.read_csv(self.type + "_survival.tsv", '\t')
        raw_input = df.values[:, 1:]
        self.X['sur'] = np.array(raw_input)

        df = pd.read_csv(self.type + "_censored.tsv", '\t')
        raw_input = df.values[:, 1:]
        self.X['cen'] = np.array(raw_input)

    def data_preprocess(self):
        for i in range(self.X['mRNA'].shape[0]):
            min, max = np.min(self.X['mRNA'][i, :]), np.max(self.X['mRNA'][i, :])
            if max - min != 0:
                for j in range(self.X['mRNA'].shape[1]):
                    self.X['mRNA'][i, j] = (self.X['mRNA'][i, j] - min) / (max - min)

        for i in range(self.X['cen'].shape[1]):
            self.X['cen'][0, i] = 1 - self.X['cen'][0, i]

        '''
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])
        for i in range(self.Y.shape[0]):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv
        '''

    def data_split(self):
        all = np.split(self.X['all'], [self.train, self.valid, self.samples], axis=1)
        cli = np.split(self.X['cli'], [self.train, self.valid, self.samples], axis=1)
        mut = np.split(self.X['mut'], [self.train, self.valid, self.samples], axis=1)
        cnv = np.split(self.X['CNV'], [self.train, self.valid, self.samples], axis=1)
        mrna = np.split(self.X['mRNA'], [self.train, self.valid, self.samples], axis=1)
        sur = np.split(self.X['sur'], [self.train, self.valid, self.samples], axis=1)
        cen = np.split(self.X['cen'], [self.train, self.valid, self.samples], axis=1)
        self.T = {'all': all[0], 'cli': cli[0], 'mut': mut[0], 'CNV': cnv[0],
                  'mRNA': mrna[0], 'sur': sur[0], 'cen': cen[0]}
        self.V = {'all': all[1], 'cli': cli[1], 'mut': mut[1], 'CNV': cnv[1],
                  'mRNA': mrna[1], 'sur': sur[1], 'cen': cen[1]}
        self.S = {'all': all[2], 'cli': cli[2], 'mut': mut[2], 'CNV': cnv[2],
                  'mRNA': mrna[2], 'sur': sur[2], 'cen': cen[2]}

    def data_rearrange(self):
        temp = np.argsort(self.T['sur'])[0]
        for t in self.T:
            self.T[t] = self.T[t][:, temp]
        temp = np.argsort(self.V['sur'])[0]
        for v in self.V:
            self.V[v] = self.V[v][:, temp]
        temp = np.argsort(self.S['sur'])[0]
        for s in self.S:
            self.S[s] = self.S[s][:, temp]








