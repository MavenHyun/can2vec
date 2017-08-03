import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()

sa1 = sa.split_ae(tr.x_cli, 10, 0.1, False)
sa1_enc1 = sa1.construct_encoder("Encoding_Layer1", sa1.input_data, sa1.X_train.shape[1], 10, 0)
sa1_enc2 = sa1.construct_encoder("Encoding_Layer2", sa1_enc1.result, 10, 10, 0)
sa1_dec2 = sa1.construct_decoder("Decoding_Layer2", sa1_enc2.result, 10, 10, 0)
sa1_dec1 = sa1.construct_decoder("Decoding_Layer1", sa1_dec2.result, 10, sa1.X_train.shape[1], 0)
sa1_opti = sa1.construct_optimizer("Optimizer", sa1_dec1.result, sa1.input_data, 0.1, 0)
sa1.initiate(sa1_enc1.weights, sa1_enc1.bias)










'''

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
    sa = [sa1, sa2, sa3, sa4]


maven = ma.multi_ae(tr, sa1, sa2, sa3, sa4, 0.001)
maven.initiate()
'''
