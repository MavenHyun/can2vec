import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 40, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()

for iter in range(10):
    sa1 = sa.split_ae(tr.x_cli, tr.X_cli_eval, tr.X_cli_test, 10, 0.1)
    sa1.initiate()
    sa2 = sa.split_ae(tr.x_mut, tr.X_mut_eval, tr.X_mut_test, 20, 0.1)
    sa2.initiate()
    sa3 = sa.split_ae(tr.x_CNV, tr.X_CNV_eval, tr.X_CNV_test, 500, 0.1)
    sa3.initiate()
    sa4 = sa.split_ae(tr.x_mRNA, tr.X_mRNA_eval, tr.X_mRNA_test, 1000, 0.1)
    sa4.initiate()
    sa1.printout()
    sa2.printout()
    sa3.printout()
    sa4.printout()

'''
maven = ma.multi_ae(tr, sa1, sa2, sa3, sa4, 0.001)
maven.initiate()
'''
