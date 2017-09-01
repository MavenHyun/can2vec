import numpy as np
import pandas as pd
import statistics as st

class data_set:
    def __init__(self, cancer_type, num_train, num_valid):
        self.type = cancer_type
        self.train = num_train
        self.valid = num_valid
        self.samples = 0

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
                  'mRNA': np.array(df.values[c + m + v:c + m + v + r, 1:]).transpose()}
        self.samples = self.X['all'].shape[0]
        
        df = pd.read_csv(self.type + "_survival.tsv", '\t')
        raw_input = df.values[:, 1:]
        self.X['sur'] = np.array(raw_input).transpose()

        df = pd.read_csv(self.type + "_censored.tsv", '\t')
        raw_input = df.values[:, 1:]
        self.X['cen'] = np.array(raw_input).transpose()

    def data_preprocess(self):
        for i in range(self.X['mRNA'].shape[1]):
            min, max = np.min(self.X['mRNA'][:, i]), np.max(self.X['mRNA'][:, i])
            if max - min != 0:
                for j in range(self.X['mRNA'].shape[0]):
                    self.X['mRNA'][j, i] = (self.X['mRNA'][j, i] - min) / (max - min)

        for i in range(self.X['cen'].shape[0]):
            self.X['cen'][i, 0] = 1 - self.X['cen'][i, 0]

        '''
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])
        for i in range(self.Y.shape[0]):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv
        '''

    def data_split(self):
        all = np.split(self.X['all'], [self.train, self.valid, self.samples], axis=0)
        cli = np.split(self.X['cli'], [self.train, self.valid, self.samples], axis=0)
        mut = np.split(self.X['mut'], [self.train, self.valid, self.samples], axis=0)
        cnv = np.split(self.X['CNV'], [self.train, self.valid, self.samples], axis=0)
        mrna = np.split(self.X['mRNA'], [self.train, self.valid, self.samples], axis=0)
        sur = np.split(self.X['sur'], [self.train, self.valid, self.samples], axis=0)
        cen = np.split(self.X['sur'], [self.train, self.valid, self.samples], axis=0)
        self.T = {'all': all[0], 'cli': cli[0], 'mut': mut[0], 'CNV': cnv[0],
                  'mRNA': mrna[0], 'sur': sur[0], 'cen': cen[0]}
        self.V = {'all': all[1], 'cli': cli[1], 'mut': mut[1], 'CNV': cnv[1],
                  'mRNA': mrna[1], 'sur': sur[1], 'cen': cen[1]}
        self.S = {'all': all[2], 'cli': cli[2], 'mut': mut[2], 'CNV': cnv[2],
                  'mRNA': mrna[2], 'sur': sur[2], 'cen': cen[2]}
