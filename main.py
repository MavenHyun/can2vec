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

enc = maven.top_encoder('cli', 100, 'relu')
cli_T = maven.bot_decoder(enc, 'cli', 100, 'relu')
maven.mirror_image('cli', cli_T, maven.P['cli'], 'adam', 50001, 1e-4)
cli = maven.master_encoder('cli', 'relu')

enc = maven.top_encoder('mut', 100, 'relu')
mut_T = maven.bot_decoder(enc, 'mut', 100, 'relu')
maven.mirror_image('mut', mut_T, maven.P['mut'], 'adam', 50001, 1e-4)
mut = maven.master_encoder('mut', 'relu')

enc = maven.top_encoder('CNV', 1000, 'relu')
enc1 = maven.mid_encoder('CNV', 1000, 500, 'relu', enc)
enc2 = maven.mid_encoder('CNV', 500, 250, 'relu', enc1)
enc3 = maven.mid_encoder('CNV', 250, 100, 'relu', enc2)
dec3 = maven.mid_encoder('CNV', 100, 250, 'relu', enc3)
dec2 = maven.mid_encoder('CNV', 250, 500, 'relu', dec3)
dec1 = maven.mid_encoder('CNV', 500, 1000, 'relu', dec2)
CNV_T = maven.bot_decoder(dec1, 'CNV', 1000, 'relu')
maven.mirror_image('CNV', CNV_T, maven.P['CNV'], 'adam', 50001, 1e-4)
CNV = maven.master_encoder('CNV', 'relu')

enc = maven.top_encoder('mRNA', 10000, 'relu')
enc1 = maven.mid_encoder('mRNA', 10000, 5000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA', 5000, 2500, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA', 2500, 1000, 'relu', enc2)
enc4 = maven.mid_encoder('mRNA', 1000, 500, 'relu', enc3)
enc5 = maven.mid_encoder('mRNA', 500, 200, 'relu', enc4)
enc6 = maven.mid_encoder('mRNA', 250, 100, 'relu', enc5)
dec6 = maven.mid_encoder('mRNA', 100, 250, 'relu', enc6)
dec5 = maven.mid_encoder('mRNA', 250, 500, 'relu', dec6)
dec4 = maven.mid_encoder('mRNA', 500, 1000, 'relu', dec5)
dec3 = maven.mid_encoder('mRNA', 1000, 2500, 'relu', dec4)
dec2 = maven.mid_encoder('mRNA', 2500, 5000, 'relu', dec3)
dec1 = maven.mid_encoder('mRNA', 5000, 10000, 'relu', dec2)
mRNA_T = maven.bot_decoder(dec1, 'mRNA', 10000, 'relu')
maven.mirror_image('mRNA', mRNA_T, maven.P['mRNA'], 'adam', 10001, 0.1)
mRNA = maven.master_encoder('mRNA', 'relu')

vector = tf.concat([cli, mut, CNV, mRNA], 1)

pro1 = maven.data_projector(vector, 400, 400, 'relu')
pro2 = maven.data_projector(pro1, 400, 400, 'relu')
pred = maven.surv_predictor(pro2, 400, 'relu')
maven.foresight(pred, maven.P['surviv'], 'adag', 10001, 1e-6)








