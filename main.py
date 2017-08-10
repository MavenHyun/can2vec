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
maven.mirror_image('CNV', CNV_T, maven.P['CNV'], 'adam', 50001, 1e-4)
CNV = maven.master_encoder('CNV', 'tanh')

enc = maven.top_encoder('mRNA', 10000, 'relu')
enc1 = maven.mid_encoder('mRNA', 10000, 5000, 'relu', enc)
enc2 = maven.mid_encoder('mRNA', 5000, 1000, 'relu', enc1)
enc3 = maven.mid_encoder('mRNA', 1000, 500, 'relu', enc2)
enc4 = maven.mid_encoder('mRNA', 500, 100, 'relu', enc3)
dec4 = maven.mid_decoder('mRNA', 100, 500, 'relu', enc4)
dec3 = maven.mid_encoder('mRNA', 500, 1000, 'relu', dec4)
dec2 = maven.mid_decoder('mRNA', 1000, 5000, 'relu', dec3)
dec1 = maven.mid_decoder('mRNA', 5000, 10000, 'relu', dec2)
mRNA_T = maven.bot_decoder(dec1, 'mRNA', 10000, 'relu')
maven.mirror_image('mRNA', mRNA_T, maven.P['mRNA'], 'adam', 50001, 1e-4)
mRNA = maven.master_encoder('mRNA', 'relu')

vector = tf.concat([mut, CNV, mRNA], 1)

pro1 = maven.data_projector(vector, 300, 300, 'relu')
pro2 = maven.data_projector(pro1, 300, 300, 'relu')
pred = maven.surv_predictor(pro2, 300, 'relu')
maven.foresight(pred, maven.P['surviv'], 'adag', 10001, 1e-6)








