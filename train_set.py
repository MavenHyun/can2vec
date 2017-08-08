import numpy as np
import pandas as pd
import statistics as st

class data_set:
    def __init__(self, f, s, n1, n2):
        self.file_name, self.file_name2, self.size_eval, self.size_test = f, s, n1, n1 + n2

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

        self.X = {'all': np.array(df.values[:c + m + v + r, 1:]).transpose(),
                  'cli': np.array(df.values[:c, 1:]).transpose(),
                  'mut': np.array(df.values[c:c + m, 1:]).transpose(),
                  'CNV': np.array(df.values[c + m:c + m + v, 1:]).transpose(),
                  'mRNA': np.array(df.values[c + m + v:c + m + v + r, 1:]).transpose()}

        df = pd.read_csv(self.file_name2, '\t')
        raw_input = df.values[:, 1:]
        self.Y = np.array(raw_input).transpose()

    def data_preprocess(self):
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


