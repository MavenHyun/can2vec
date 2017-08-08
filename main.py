import multi_ae as ma
import split_ae as sa
import train_set as ts
import c2v_models as cv

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
tr.data_extract()
tr.data_preprocess()


#for activation, 0->relu, 1->sigmoid, 2->tanh, other->nothing
#for optimization, 0->Adam, 1->RMS, 2->Adagrad, 3->Adadelta, other->GradientDescent






maven = cv.NoviceSeer(tr, 50, 80)
enc = maven.leading_encoder('cli', 10, 'relu')
pre = maven.surv_predictor(enc, 10, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adam', 5001, 0.01)


enc = maven.leading_encoder('mut', 40, 'relu')
pre = maven.surv_predictor(enc, 40, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adam', 5001, 0.01)










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
