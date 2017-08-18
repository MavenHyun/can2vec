import train_set as ts
import c2v_models as cv
import tensorflow as tf


def run_model(pretrain):
    tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
    tr.data_extract()
    tr.data_preprocess()
    maven = cv.FarSeer(tr, 45, 30, 0.666)

    with tf.name_scope("Cli_AEncoder"):
        cli = maven.not_encoder('cli')

    with tf.name_scope("mRNA_AEncoder"):
        enc = maven.top_encoder('mRNA', 5000, 'relu')
        enc1 = maven.mid_encoder('mRNA', 5000, 4000, 'relu', enc)
        enc2 = maven.mid_encoder('mRNA', 4000, 1000, 'relu', enc1)
        enc3 = maven.mid_encoder('mRNA', 1000, 400, 'relu', enc2)
        dec3 = maven.mid_decoder('mRNA', 400, 1000, 'relu', enc3)
        dec2 = maven.mid_decoder('mRNA', 1000, 4000, 'relu', dec3)
        dec1 = maven.mid_decoder('mRNA', 4000, 5000, 'relu', dec2)
        mRNA_T = maven.bot_decoder(dec1, 'mRNA', 5000, 'relu')
        mRNA = enc3

    with tf.name_scope("mut_AEncoder"):
        enc = maven.top_encoder('mut', 100, 'relu')
        mut_T = maven.bot_decoder(enc, 'mut', 100, 'relu')
        mut = enc

    with tf.name_scope("CNV_AEncoder"):
        enc = maven.top_encoder('CNV', 1000, 'tanh')
        enc1 = maven.mid_encoder('CNV', 1000, 100, 'tanh', enc)
        dec1 = maven.mid_decoder('CNV', 100, 1000, 'tanh', enc1)
        CNV_T = maven.bot_decoder(dec1, 'CNV', 1000, 'tanh')
        CNV = enc1

    if pretrain is True:
        maven.item_list.append(cv.SplitOptimizer('mRNA', mRNA_T, maven.P['mRNA'], 'adam', 1001, 1e-3))
        maven.item_list.append(cv.SplitOptimizer('mut', mut_T, maven.P['mut'], 'adam', 1001, 1e-3))
        maven.item_list.append(cv.SplitOptimizer('CNV', CNV_T, maven.P['CNV'], 'adam', 1001, 1e-3))
        maven.optimize_AEncoders()

    with tf.name_scope("Feature_Vector"):
        vector = tf.concat([mRNA, CNV, mut, cli], 1)

    with tf.name_scope("Survivability_Prediction"):
        pro = maven.data_projector(vector, 619, 619, 'relu')
        pro2 = maven.data_projector(pro, 619, 619, 'relu')
        pre = maven.surv_predictor(pro2, 619, 'relu')
        maven.optimize_SPredictor(pre, maven.P['surviv'], 'adag', 10001, 1e-3)