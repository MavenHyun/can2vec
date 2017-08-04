import multi_ae as ma
import split_ae as sa
import train_set as ts

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()

#for activation, 0->relu, 1->sigmoid, 2->tanh, other->nothing
#for optimization, 0->Adam, 1->RMS, 2->Adagrad, 3->Adadelta, other->GradientDescent


sae_cli = sa.split_ae(tr.x_cli, False)
sae_cli_enc1 = sae_cli.construct_Encoder("cli_Encoding_Layer1", sae_cli.input_data, sae_cli.F_num, 10, 1)
sae_cli_dec1 = sae_cli.construct_Decoder("cli_Decoding_Layer1", sae_cli_enc1.result, 10, sae_cli.F_num, 1)
sae_cli_opti = sae_cli.construct_Optimizer("cli_Optimizer", sae_cli_dec1.result, sae_cli.input_data, 0.1, 0)
product_cli = sae_cli.initiate(sae_cli_opti, 500, sae_cli_enc1)

sae_mut = sa.split_ae(tr.x_mut, False)
sae_mut_enc1 = sae_mut.construct_Encoder("mut_Encoding_Layer1", sae_mut.input_data, sae_mut.F_num, 40, 1)
sae_mut_dec1 = sae_mut.construct_Decoder("mut_Decoding_Layer1", sae_mut_enc1.result, 40, sae_mut.F_num, 1)
sae_mut_opti = sae_mut.construct_Optimizer("mut_Optimizer", sae_mut_dec1.result, sae_mut.input_data, 0.1, 0)
product_mut = sae_mut.initiate(sae_mut_opti, 500, sae_mut_enc1)

sae_CNV = sa.split_ae(tr.x_CNV, False)
sae_CNV_enc1 = sae_CNV.construct_Encoder("CNV_Encoding_Layer1", sae_CNV.input_data, sae_CNV.F_num, 500, 0)
sae_CNV_enc2 = sae_CNV.construct_Encoder("CNV_Encoding_Layer2", sae_CNV_enc1.result, 500, 250, 0)
sae_CNV_dec2 = sae_CNV.construct_Decoder("CNV_Decoding_Layer2", sae_CNV_enc2.result, 250, 500, 0)
sae_CNV_dec1 = sae_CNV.construct_Decoder("CNV_Decoding_Layer1", sae_CNV_dec2.result, 500, sae_CNV.F_num, 0)
sae_CNV_opti = sae_CNV.construct_Optimizer("CNV_Optimizer", sae_CNV_dec1.result, sae_CNV.input_data, 0.3, 0)
product_CNV = sae_CNV.initiate(sae_CNV_opti, 500, sae_CNV_enc1)

sae_mRNA = sa.split_ae(tr.x_mRNA, False)
sae_mRNA_enc1 = sae_mRNA.construct_Encoder("mRNA_Encoding_Layer1", sae_mRNA.input_data, sae_mRNA.F_num, 800, 0)
sae_mRNA_enc2 = sae_mRNA.construct_Encoder("mRNA_Encoding_Layer2", sae_mRNA_enc1.result, 800, 550, 0)
sae_mRNA_enc3 = sae_mRNA.construct_Encoder("mRNA_Encoding_Layer3", sae_mRNA_enc2.result, 550, 300, 0)
sae_mRNA_dec3 = sae_mRNA.construct_Decoder("mRNA_Decoding_Layer3", sae_mRNA_enc3.result, 300, 550, 0)
sae_mRNA_dec2 = sae_mRNA.construct_Decoder("mRNA_Decoding_Layer2", sae_mRNA_dec3.result, 550, 800, 0)
sae_mRNA_dec1 = sae_mRNA.construct_Decoder("mRNA_Decoding_Layer1", sae_mRNA_dec2.result, 800, sae_mRNA.F_num, 0)
sae_mRNA_opti = sae_mRNA.construct_Optimizer("mRNA_Optimizer", sae_mRNA_dec1.result, sae_mRNA.input_data, 0.4, 0)
product_mRNA = sae_mRNA.initiate(sae_mRNA_opti, 500, sae_mRNA_enc1)

total_hidden_nodes = 600

sae_cli.print_result(sae_cli_opti)
sae_mut.print_result(sae_mut_opti)
sae_CNV.print_result(sae_CNV_opti)
sae_mRNA.print_result(sae_mRNA_opti)

maven = ma.multi_ae(tr, product_cli, product_mut, product_CNV, product_mRNA, total_hidden_nodes)

maven_projector = maven.construct_Projector("Projection_Layer", 0)
maven_predictor = maven.construct_Predictor("Prediction_Layer", maven_projector.result, 0)
maven_pre_optimizer = maven.construct_Optimizer("Optimizer_S", maven_predictor.result, maven.inputs['answer_S'], 0.1, 0)
maven.initiate(maven_pre_optimizer, 1000)















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
