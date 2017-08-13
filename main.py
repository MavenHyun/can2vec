import multi_ae as ma
import split_ae as sa
import train_set as ts
import c2v_models as cv
import tensorflow as tf

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
tr.data_extract()
tr.data_preprocess()

'''
maven = cv.SeerAdept(tr, 45, 30)
ae = maven.training_autoencoder('cli', 19, 'relu')
maven.mirror_image(ae, maven.P['cli'], 'adam', 50001, 0.01)
le = maven.leading_encoder('cli', 'relu')
pro1 = maven.data_projector1(le, 19, 400, 'relu')
pro2 = maven.data_projector1(pro1, 400, 19, 'relu')
pre = maven.surv_predictor(pro2, 19, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adag', 500001, 0.1)

'''

maven = cv.FarSeer(tr, 45, 30, 0.666)
'''
enc = maven.top_encoder('cli', 100, 'relu')
cli_T = maven.bot_decoder(enc, 'cli', 100, 'relu')
maven.mirror_image('cli', cli_T, maven.P['cli'], 'adam', 50001, 1e-4)
cli = maven.master_encoder('cli', 'relu')
'''

enc = maven.top_encoder('mut', 100, 'relu')
mut_T = maven.bot_decoder(enc, 'mut', 100, 'relu')
maven.mirror_image('mut', mut_T, maven.P['mut'], 'adam', 50001, 1e-4)
mut = maven.master_encoder('mut', 'relu')

enc = maven.top_encoder('CNV', 1000, 'tanh')
enc1 = maven.mid_encoder('CNV', 1000, 500, 'tanh', enc)
enc2 = maven.mid_encoder('CNV', 500, 100, 'tanh', enc1)
dec2 = maven.mid_decoder('CNV', 100, 500, 'tanh', enc2)
dec1 = maven.mid_decoder('CNV', 500, 1000, 'tanh', dec2)
CNV_T = maven.bot_decoder(dec1, 'CNV', 1000, 'tanh')
maven.mirror_image('CNV', CNV_T, maven.P['CNV'], 'adam', 50001, 1e-3)
CNV = maven.master_encoder('CNV', 'tanh')

enc = maven.top_encoder('mRNA1', 2000, 'relu')
enc1 = maven.mid_encoder('mRNA1', 2000, 1000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA1', 1000, 500, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA1', 500, 100, 'relu', enc2)
dec3 = maven.mid_decoder('mRNA1', 100, 500, 'relu', enc3)
dec2 = maven.mid_decoder('mRNA1', 500, 1000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA1', 1000, 2000, 'relu', dec2)
mRNA1_T = maven.bot_decoder(dec1, 'mRNA1', 2000, 'relu')
maven.mirror_image('mRNA1', mRNA1_T, maven.P['mRNA1'], 'adam', 50001, 1e-6)
mRNA1 = maven.master_encoder('mRNA1', 'relu')

enc = maven.top_encoder('mRNA2', 2000, 'relu')
enc1 = maven.mid_encoder('mRNA2', 2000, 1000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA2', 1000, 500, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA2', 500, 100, 'relu', enc2)
dec3 = maven.mid_decoder('mRNA2', 100, 500, 'relu', enc3)
dec2 = maven.mid_decoder('mRNA2', 500, 1000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA2', 1000, 2000, 'relu', dec2)
mRNA2_T = maven.bot_decoder(dec1, 'mRNA2', 2000, 'relu')
maven.mirror_image('mRNA2', mRNA2_T, maven.P['mRNA2'], 'adam', 50001, 1e-6)
mRNA2 = maven.master_encoder('mRNA2', 'relu')

enc = maven.top_encoder('mRNA3', 2000, 'relu')
enc1 = maven.mid_encoder('mRNA3', 2000, 1000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA3', 1000, 500, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA3', 500, 100, 'relu', enc2)
dec3 = maven.mid_decoder('mRNA3', 100, 500, 'relu', enc3)
dec2 = maven.mid_decoder('mRNA3', 500, 1000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA3', 1000, 2000, 'relu', dec2)
mRNA3_T = maven.bot_decoder(dec1, 'mRNA3', 2000, 'relu')
maven.mirror_image('mRNA3', mRNA3_T, maven.P['mRNA3'], 'adam', 50001, 1e-6)
mRNA3 = maven.master_encoder('mRNA3', 'relu')

enc = maven.top_encoder('mRNA4', 2000, 'relu')
enc1 = maven.mid_encoder('mRNA4', 2000, 1000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA4', 1000, 500, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA4', 500, 100, 'relu', enc2)
dec3 = maven.mid_decoder('mRNA4', 100, 500, 'relu', enc3)
dec2 = maven.mid_decoder('mRNA4', 500, 1000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA4', 1000, 2000, 'relu', dec2)
mRNA4_T = maven.bot_decoder(dec1, 'mRNA4', 2000, 'relu')
maven.mirror_image('mRNA4', mRNA4_T, maven.P['mRNA4'], 'adam', 50001, 1e-6)
mRNA4 = maven.master_encoder('mRNA4', 'relu')

vector = tf.concat([mut, CNV, mRNA1, mRNA2, mRNA3, mRNA3], 1)


pro1 = maven.data_projector(vector, 600, 600, 'relu')
pro2 = maven.data_projector(pro1, 600, 600, 'relu')
pred = maven.surv_predictor(pro2, 600, 'relu')
maven.foresight(pred, maven.P['surviv'], 'adag', 10001, 1e-6)










