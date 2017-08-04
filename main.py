import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()

#for activation, 0->relu, 1->sigmoid, 2->tanh, other->nothing
#for optimization, 0->Adam, 1->RMS, 2->Adagrad, 3->Adadelta, other->GradientDescent

for iter in range(10):
    sae_cli = sa.split_ae(tr.x_cli, False)
    sae_cli_enc1 = sae_cli.construct_encoder("cli_Encoding_Layer1", sae_cli.input_data, sae_cli.F_num, 10, 1)
    sae_cli_dec1 = sae_cli.construct_decoder("cli_Decoding_Layer1", sae_cli_enc1.result, 10, sae_cli.F_num, 1)
    sae_cli_opti = sae_cli.construct_optimizer("cli_Optimizer", sae_cli_dec1.result, sae_cli.input_data, 0.1, 0)
    sae_cli.initiate(sae_cli_opti, 1000)

    sae_mut = sa.split_ae(tr.x_mut, False)
    sae_mut_enc1 = sae_mut.construct_encoder("mut_Encoding_Layer1", sae_mut.input_data, sae_mut.F_num, 30, 1)
    sae_mut_dec1 = sae_mut.construct_decoder("mut_Decoding_Layer1", sae_mut_enc1.result, 30, sae_mut.F_num, 1)
    sae_mut_opti = sae_mut.construct_optimizer("mut_Optimizer", sae_mut_dec1.result, sae_mut.input_data, 0.1, 0)
    sae_mut.initiate(sae_mut_opti, 1000)

    sae_CNV = sa.split_ae(tr.x_CNV, False)
    sae_CNV_enc1 = sae_CNV.construct_encoder("CNV_Encoding_Layer1", sae_CNV.input_data, sae_CNV.F_num, 500, 0)
    sae_CNV_enc2 = sae_CNV.construct_encoder("CNV_Encoding_Layer2", sae_CNV_enc1.result, 500, 200, 0)
    sae_CNV_dec2 = sae_CNV.construct_decoder("CNV_Decoding_Layer2", sae_CNV_enc2.result, 200, 500, 0)
    sae_CNV_dec1 = sae_CNV.construct_decoder("CNV_Decoding_Layer1", sae_CNV_dec2.result, 500, sae_CNV.F_num, 0)
    sae_CNV_opti = sae_CNV.construct_optimizer("CNV_Optimizer", sae_CNV_dec1.result, sae_CNV.input_data, 0.3, 0)
    sae_CNV.initiate(sae_CNV_opti, 1000)

    sae_mRNA = sa.split_ae(tr.x_mRNA, False)
    sae_mRNA_enc1 = sae_mRNA.construct_encoder("mRNA_Encoding_Layer1", sae_mRNA.input_data, sae_mRNA.F_num, 800, 0)
    sae_mRNA_enc2 = sae_mRNA.construct_encoder("mRNA_Encoding_Layer2", sae_mRNA_enc1.result, 800, 650, 0)
    sae_mRNA_enc3 = sae_mRNA.construct_encoder("mRNA_Encoding_Layer3", sae_mRNA_enc2.result, 650, 300, 0)
    sae_mRNA_dec3 = sae_mRNA.construct_decoder("mRNA_Decoding_Layer3", sae_mRNA_enc3.result, 300, 650, 0)
    sae_mRNA_dec2 = sae_mRNA.construct_decoder("mRNA_Decoding_Layer2", sae_mRNA_dec3.result, 650, 800, 0)
    sae_mRNA_dec1 = sae_mRNA.construct_decoder("mRNA_Decoding_Layer1", sae_mRNA_dec2.result, 800, sae_mRNA.F_num, 0)
    sae_mRNA_opti = sae_mRNA.construct_optimizer("mRNA_Optimizer", sae_mRNA_dec1.result, sae_mRNA.input_data, 0.4, 0)
    sae_mRNA.initiate(sae_mRNA_opti, 1000)

    sae_cli.print_result(sae_cli_opti)
    sae_mut.print_result(sae_mut_opti)
    sae_CNV.print_result(sae_CNV_opti)
    sae_mRNA.print_result(sae_mRNA_opti)








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
