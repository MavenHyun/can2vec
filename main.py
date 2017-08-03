import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()

for iter in range(10):
    sa1 = sa.split_ae(tr.x_cli, 10, 0.1, False)
    sa1.initiate()
    sa2 = sa.split_ae(tr.x_mut, 30, 0.1, False)
    sa2.initiate()
    sa3 = sa.split_ae(tr.x_CNV, 500, 0.3, True)
    sa3.initiate()
    sa4 = sa.split_ae(tr.x_mRNA, 800, 0.2, True)
    sa4.initiate()
    sa1.printout()
    sa2.printout()
    sa3.printout()
    sa4.printout()

'''
maven = ma.multi_ae(tr, sa1, sa2, sa3, sa4, 0.001)
maven.initiate()
'''
