import train_set as ts
import c2v_models as cv
import tensorflow as tf

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
tr.data_extract()
tr.data_preprocess()

maven = cv.FarSeer(tr, 45, 30, 0.666)

cli = maven.D['cli']

enc = maven.top_encoder('mRNA1', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA1', 4000, 2000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA1', 2000, 1000, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA1', 1000, 500, 'relu', enc2)
enc4 = maven.mid_encoder('mRNA1', 500, 100, 'relu', enc3)
dec4 = maven.mid_decoder('mRNA1', 100, 500, 'relu', enc4)
dec3 = maven.mid_decoder('mRNA1', 500, 1000, 'relu', dec4)
dec2 = maven.mid_decoder('mRNA1', 1000, 2000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA1', 2000, 4000, 'relu', dec2)
mRNA1_T = maven.bot_decoder(dec1, 'mRNA1', 4000, 'relu')
maven.mirror_image('mRNA1', mRNA1_T, maven.P['mRNA1'], 'adam', 10001, 9e-2)
mRNA1 = enc4

enc = maven.top_encoder('mRNA2', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA2', 4000, 2000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA2', 2000, 1000, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA2', 1000, 500, 'relu', enc2)
enc4 = maven.mid_encoder('mRNA2', 500, 100, 'relu', enc3)
dec4 = maven.mid_decoder('mRNA2', 100, 500, 'relu', enc4)
dec3 = maven.mid_decoder('mRNA2', 500, 1000, 'relu', dec4)
dec2 = maven.mid_decoder('mRNA2', 1000, 2000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA2', 2000, 4000, 'relu', dec2)
mRNA2_T = maven.bot_decoder(dec1, 'mRNA2', 4000, 'relu')
maven.mirror_image('mRNA2', mRNA2_T, maven.P['mRNA2'], 'adam', 10001, 9e-2)
mRNA2 = enc4

enc = maven.top_encoder('mRNA3', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA3', 4000, 2000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA3', 2000, 1000, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA3', 1000, 500, 'relu', enc2)
enc4 = maven.mid_encoder('mRNA3', 500, 100, 'relu', enc3)
dec4 = maven.mid_decoder('mRNA3', 100, 500, 'relu', enc4)
dec3 = maven.mid_decoder('mRNA3', 500, 1000, 'relu', dec4)
dec2 = maven.mid_decoder('mRNA3', 1000, 2000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA3', 2000, 4000, 'relu', dec2)
mRNA3_T = maven.bot_decoder(dec1, 'mRNA3', 4000, 'relu')
maven.mirror_image('mRNA3', mRNA3_T, maven.P['mRNA3'], 'adam', 10001, 9e-2)
mRNA3 = enc4

enc = maven.top_encoder('mRNA4', 4000, 'relu')
enc1 = maven.mid_encoder('mRNA4', 4000, 2000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA4', 2000, 1000, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA4', 1000, 500, 'relu', enc2)
enc4 = maven.mid_encoder('mRNA4', 500, 100, 'relu', enc3)
dec4 = maven.mid_decoder('mRNA4', 100, 500, 'relu', enc4)
dec3 = maven.mid_decoder('mRNA4', 500, 1000, 'relu', dec4)
dec2 = maven.mid_decoder('mRNA4', 1000, 2000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA4', 2000, 4000, 'relu', dec2)
mRNA4_T = maven.bot_decoder(dec1, 'mRNA4', 4000, 'relu')
maven.mirror_image('mRNA4', mRNA4_T, maven.P['mRNA4'], 'adam', 10001, 9e-2)
mRNA4 = enc4

enc = maven.top_encoder('mut', 100, 'relu')
mut_T = maven.bot_decoder(enc, 'mut', 100, 'relu')
maven.mirror_image('mut', mut_T, maven.P['mut'], 'adam', 10001, 1e-3)
mut = enc

enc = maven.top_encoder('CNV', 1000, 'tanh')
enc1 = maven.mid_encoder('CNV', 1000, 100, 'tanh', enc)
dec1 = maven.mid_decoder('CNV', 100, 1000, 'tanh', enc1)
CNV_T = maven.bot_decoder(dec1, 'CNV', 1000, 'tanh')
maven.mirror_image('CNV', CNV_T, maven.P['CNV'], 'adam', 10001, 1e-3)
CNV = enc1
vector = tf.concat([cli, mut, CNV, mRNA1, mRNA2, mRNA3, mRNA3], 1)

pro = maven.the_alchemist('Survivability', vector, 619, 619, 'relu')
pre = maven.surv_predictor(pro, 619, 'relu')
maven.foresight(pre, maven.S, 'adag', 10001, 1e-3)









