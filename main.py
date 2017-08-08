import multi_ae as ma
import split_ae as sa
import train_set as ts
import c2v_models as cv

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
tr.data_extract()
tr.data_preprocess()
maven = cv.NoviceSeer(tr, 45, 30)
enc = maven.leading_encoder('cli', 10, 'relu')
pro = maven.data_projector(enc, 10, 'relu')
pre = maven.surv_predictor(pro, 10, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adam', 50001, 0.9)

maven = cv.NoviceSeer(tr, 45, 30)
enc = maven.leading_encoder('mut', 40, 'relu')
pro = maven.data_projector(enc, 40, 'relu')
pre = maven.surv_predictor(enc, 40, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adam', 50001, 0.9)

maven = cv.NoviceSeer(tr, 45, 30)
enc = maven.leading_encoder('CNV', 800, 'relu')
pro = maven.data_projector(enc, 800, 'relu')
pre = maven.surv_predictor(enc, 800, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adam', 50001, 0.9)

maven = cv.NoviceSeer(tr, 45, 30)
enc = maven.leading_encoder('mRNA', 4000, 'relu')
pro = maven.data_projector(enc, 4000, 'relu')
pre = maven.surv_predictor(enc, 4000, 'relu')
maven.foresight(pre, maven.P['surviv'], 'adam', 50001, 0.9)
