import multi_ae as ma


tr = ma.train_set('ACC_features.tsv', 'ACC_survival.tsv')
tr.data_preprocess()
maven = ma.multi_ae(tr, 10, 20, 100, 800, 0.001)
maven.execute(tr)


