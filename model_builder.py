import train_set as ts
import c2v_models as cv
import tensorflow as tf

def create_model(pretrain):
    tr = ts.data_set("ACC", 45, 30)
    tr.data_extract()
    tr.data_preprocess()
    tr.data_split()
    tr.data_rearrange()
    maven = cv.FarSeer(tr, 0.888)

    with tf.name_scope("Cli_AEncoder"):
        cli = maven.not_encoder('cli')

    with tf.name_scope("mRNA_AEncoder"):
        enc = maven.top_encoder('mRNA', 5000, 'relu')
        enc1 = maven.mid_encoder('mRNA', 2000, 'relu', enc)
        enc2 = maven.mid_encoder('mRNA', 400, 'relu', enc1)
        dec2 = maven.mid_decoder('mRNA', 2000, 'relu', enc2)
        dec1 = maven.mid_decoder('mRNA', 5000, 'relu', dec2)
        mRNA_T = maven.bot_decoder('mRNA', 'relu', dec1)
        mRNA = enc2

    with tf.name_scope("mut_AEncoder"):
        enc = maven.top_encoder('mut', 100, 'relu')
        mut_T = maven.bot_decoder('mut', 'relu', enc)
        mut = enc

    with tf.name_scope("CNV_AEncoder"):
        enc = maven.top_encoder('CNV', 1000, 'relu')
        enc1 = maven.mid_encoder('CNV', 100, 'relu', enc)
        dec1 = maven.mid_decoder('CNV', 1000, 'relu', enc1)
        CNV_T = maven.bot_decoder('CNV', 'relu', dec1)
        CNV = enc1

    if pretrain is True:
        maven.item_list.append(cv.SplitOptimizer('mRNA', mRNA_T, maven.P['mRNA'], 'adam', 50001, 1e-3))
        maven.item_list.append(cv.SplitOptimizer('mut', mut_T, maven.P['mut'], 'adam', 50001, 1e-3))
        maven.item_list.append(cv.SplitOptimizer('CNV', CNV_T, maven.P['CNV'], 'adam', 50001, 1e-3))
        maven.optimize_AEncoders()

    else:
        #cli = maven.data_projector(cli, 19, 19, 'relu')
        #mRNA = maven.data_projector(mRNA, 400, 400, 'relu')
        #mut = maven.data_projector(mut, 100, 100, 'relu')
        #CNV = maven.data_projector(CNV, 100, 100, 'relu')
        vector = tf.concat([cli, mut, CNV, mRNA], 0)

        #with tf.name_scope("Survival_Prediction"):
        #   pro = maven.data_projector(vector, 619, 619, 'relu')
        #   pre = maven.surv_predictor(pro, 'relu')
        #   maven.optimize_SPredictor(pre, 'adag', 5001, 1e-3)

        with tf.name_scope("Data_Reconstruction"):
            mRNA = maven.data_projector(vector, 619, 400, 'relu')
            mRNA_dec2 = maven.mid_decoder('mRNA', 2000, 'relu', mRNA)
            mRNA_dec1 = maven.mid_decoder('mRNA', 5000, 'relu', mRNA_dec2)
            mRNA_D = maven.bot_decoder('mRNA', 'relu', mRNA_dec1)

            mut = maven.data_projector(vector, 619, 100, 'relu')
            mut_D = maven.bot_decoder('mut', 'relu', mut)

            CNV = maven.data_projector(vector, 619, 100, 'relu')
            CNV_dec1 = maven.mid_decoder('CNV', 1000, 'relu', CNV)
            CNV_D = maven.bot_decoder('CNV', 'relu', CNV_dec1)

            output = tf.concat[cli, mut_D, CNV_D, mRNA_D]
            maven.optimize_RConstructor(output, 'adam', 20001, 1e-3)

