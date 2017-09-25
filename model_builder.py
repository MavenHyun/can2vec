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
        enc = maven.top_encoder('mRNA', 10000, 'relu')
        enc1 = maven.mid_encoder('mRNA', 5000, 'relu', enc)
        dec1 = maven.mid_decoder('mRNA', 10000, 'relu', enc1)
        mRNA_T = maven.bot_decoder('mRNA', 'relu', dec1)
        mRNA = enc1

    with tf.name_scope("mut_AEncoder"):
        enc = maven.top_encoder('mut', 100, 'relu')
        mut_T = maven.bot_decoder('mut', 'relu', enc)
        mut = enc

    with tf.name_scope("CNV_AEncoder"):
        enc = maven.top_encoder('CNV', 1000, 'tanh')
        CNV_T = maven.bot_decoder('CNV', 'tanh', enc)
        CNV = enc

    if pretrain is True:
        maven.item_list.append(cv.SplitOptimizer('mRNA', mRNA_T, maven.P['mRNA'], 'adam', 5001, 1e-6))
        maven.item_list.append(cv.SplitOptimizer('mut', mut_T, maven.P['mut'], 'adam', 5001, 1e-6))
        maven.item_list.append(cv.SplitOptimizer('CNV', CNV_T, maven.P['CNV'], 'adam', 5001, 1e-6))
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
            mRNA = maven.data_projector(vector, 6119, 2000, 'relu')
            mRNA1 = maven.lesser_decoder('mRNA_decT_1', 'relu', mRNA)
            mRNA_final = maven.lesser_decoder('mRNA_decT', 'relu', mRNA1)

            mut = maven.data_projector(vector, 6119, 100, 'relu')
            mut_final = maven.lesser_decoder('mut_decT', 'relu', mut)

            CNV = maven.data_projector(vector, 6119, 9000, 'tanh')
            CNV_final = maven.lesser_decoder('CNV_decT', 'tanh', CNV)

            output = tf.concat([cli, mut_final, CNV_final, mRNA_final], 0)
            recon = maven.re_constructor(output, 'raw', False)
            maven.optimize_RConstructor(output, 'adam', 20001, 1e-3)

