import train_set as ts
import c2v_models as cv
import tensorflow as tf

def create_model(pretrain):
    tr = ts.data_set("ACC", 45, 30)
    tr.data_extract()
    tr.data_preprocess()
    tr.data_split()
    tr.data_rearrange()
    maven = cv.FarSeer(tr, 0.666)

    with tf.name_scope("Cli_AEncoder"):
        cli = maven.not_encoder('cli')

    with tf.name_scope("mRNA_AEncoder"):
        enc = maven.top_encoder('mRNA', 5000, 'relu')
        enc1 = maven.mid_encoder('mRNA', 4000, 'relu', enc)
        enc2 = maven.mid_encoder('mRNA', 400, 'relu', enc1)
        dec2 = maven.mid_decoder('mRNA', 4000, 'relu', enc2)
        dec1 = maven.mid_decoder('mRNA', 5000, 'relu', dec2)
        mRNA_T = maven.bot_decoder('mRNA', 'relu', dec1)
        mRNA = enc2

    with tf.name_scope("mut_AEncoder"):
        enc = maven.top_encoder('mut', 100, 'relu')
        mut_T = maven.bot_decoder('mut', 'relu', enc)
        mut = enc

    with tf.name_scope("CNV_AEncoder"):
        enc = maven.top_encoder('CNV', 1000, 'tanh')
        enc1 = maven.mid_encoder('CNV', 100, 'tanh', enc)
        dec1 = maven.mid_decoder('CNV', 1000, 'tanh', enc1)
        CNV_T = maven.bot_decoder('CNV', 'tanh', dec1)
        CNV = enc1

    if pretrain is True:
        maven.item_list.append(cv.SplitOptimizer('mRNA', mRNA_T, maven.P['mRNA'], 'adam', 1001, 1e-3))
        maven.item_list.append(cv.SplitOptimizer('mut', mut_T, maven.P['mut'], 'adam', 1001, 1e-3))
        maven.item_list.append(cv.SplitOptimizer('CNV', CNV_T, maven.P['CNV'], 'adam', 1001, 1e-3))
        maven.optimize_AEncoders()

    with tf.name_scope("Feature_Vector"):
        vector = tf.concat([mRNA, CNV, mut, cli], 0)

    with tf.name_scope("Survival_Prediction"):
        pro = maven.data_projector(vector, 619, 619, 'relu')
        pre = maven.surv_predictor(pro, 619, 'relu')
        maven.optimize_CPredictor(pre, 10001, 1e-15)
        maven.optimize_SPredictor(pre, 'adam', 10001, 1e-6)


