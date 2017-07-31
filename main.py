import multi_ae as ma

test = ma.multi_ae(20, 100, 500, 0.80)
test.data_training('ACC_features.tsv', 'ACC_survival.tsv')