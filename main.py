import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.train_set('ACC_features.tsv', 'ACC_survival.tsv')
tr.data_preprocess()
sa1 = sa.split_ae(tr.X_categ, 10, 0.001)
sa2 = sa.split_ae(tr.X_mut, 20, 0.001)
sa3 = sa.split_ae(tr.X_CNV, 300, 0.001)
sa4 = sa.split_ae(tr.X_mRNA, 800, 0.001)
sa1.initiate()
sa2.initiate()
sa3.initiate()
sa4.initiate()
maven = ma.multi_ae(tr, sa1, sa2, sa3, sa4, 0.001)
maven.initiate()

