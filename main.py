import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 40, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()


sa1 = sa.split_ae(tr.X_cli, tr.XX_cli, 10, 0.01)
sa1.initiate()


sa2 = sa.split_ae(tr.X_mut, tr.X_mut_eval, tr.X_mut_test, 20, 0.01)
sa2.initiate()

sa3 = sa.split_ae(tr.X_CNV, tr.XX_CNV, 300, 0.001)
sa3.initiate()


sa4 = sa.split_ae(tr.X_mRNA, tr.XX_mRNA, 800, 0.001)
sa4.initiate()




'''
maven = ma.multi_ae(tr, sa1, sa2, sa3, sa4, 0.001)
maven.initiate()
'''
