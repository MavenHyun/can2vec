import multi_ae as ma
import split_ae as sa
import train_set as ts
import c2v_models as cv
import tensorflow as tf

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
tr.data_extract()
tr.data_preprocess()

maven = cv.FarSeer(tr, 45, 30, 0.666)

cli = maven.D['cli']

enc = maven.top_encoder('mRNA1', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA1', 4000, 800, 'relu', enc)
enc2 = maven.mid_encoder('mRNA1', 800, 100, 'relu', enc1)
dec2 = maven.mid_decoder('mRNA1', 100, 800, 'relu', enc2)
dec1 = maven.mid_decoder('mRNA1', 800, 4000, 'relu', dec2)
mRNA1_T = maven.bot_decoder(dec1, 'mRNA1', 4000, 'relu')
maven.mirror_image('mRNA1', mRNA1_T, maven.P['mRNA1'], 'adam', 5001, 1e-2)
mRNA1 = maven.master_encoder('mRNA1', 'relu')

enc = maven.top_encoder('mRNA2', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA2', 4000, 800, 'relu', enc)
enc2 = maven.mid_encoder('mRNA2', 800, 100, 'relu', enc1)
dec2 = maven.mid_decoder('mRNA2', 100, 800, 'relu', enc2)
dec1 = maven.mid_decoder('mRNA2', 800, 4000, 'relu', dec2)
mRNA2_T = maven.bot_decoder(dec1, 'mRNA2', 4000, 'relu')
maven.mirror_image('mRNA2', mRNA2_T, maven.P['mRNA2'], 'adam', 5001, 1e-2)
mRNA2 = maven.master_encoder('mRNA2', 'relu')

enc = maven.top_encoder('mRNA3', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA3', 4000, 800, 'relu', enc)
enc2 = maven.mid_encoder('mRNA3', 800, 100, 'relu', enc1)
dec2 = maven.mid_decoder('mRNA3', 100, 800, 'relu', enc2)
dec1 = maven.mid_decoder('mRNA3', 800, 4000, 'relu', dec2)
mRNA3_T = maven.bot_decoder(dec1, 'mRNA3', 4000, 'relu')
maven.mirror_image('mRNA3', mRNA3_T, maven.P['mRNA3'], 'adam', 5001, 1e-2)
mRNA3 = maven.master_encoder('mRNA3', 'relu')

enc = maven.top_encoder('mRNA4', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA4', 4000, 800, 'relu', enc)
enc2 = maven.mid_encoder('mRNA4', 800, 100, 'relu', enc1)
dec2 = maven.mid_decoder('mRNA4', 100, 800, 'relu', enc2)
dec1 = maven.mid_decoder('mRNA4', 800, 4000, 'relu', dec2)
mRNA4_T = maven.bot_decoder(dec1, 'mRNA4', 4000, 'relu')
maven.mirror_image('mRNA4', mRNA4_T, maven.P['mRNA4'], 'adam', 5001, 1e-2)
mRNA4 = maven.master_encoder('mRNA4', 'relu')

enc = maven.top_encoder('mut', 100, 'relu')
mut_T = maven.bot_decoder(enc, 'mut', 100, 'relu')
maven.mirror_image('mut', mut_T, maven.P['mut'], 'adam', 5001, 1e-4)
mut = maven.master_encoder('mut', 'relu')

enc = maven.top_encoder('CNV', 1000, 'tanh')
enc1 = maven.mid_encoder('CNV', 1000, 100, 'tanh', enc)
dec1 = maven.mid_decoder('CNV', 100, 1000, 'tanh', enc1)
CNV_T = maven.bot_decoder(dec1, 'CNV', 1000, 'tanh')
maven.mirror_image('CNV', CNV_T, maven.P['CNV'], 'adam', 5001, 1e-3)
CNV = maven.master_encoder('CNV', 'tanh')

vector = tf.concat([mut, CNV, mRNA1, mRNA2, mRNA3, mRNA3], 1)

pro1 = maven.data_projector(vector, 600, 600, 'relu')
pro2 = maven.data_projector(pro1, 600, 600, 'relu')
pred = maven.surv_predictor(pro2, 600, 'relu')
maven.foresight(pred, maven.P['surviv'], 'adag', 10001, 1e-6)










