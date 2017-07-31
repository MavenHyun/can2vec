import multi_ae as ma


tr = ma.train_set('ACC_features.tsv', 'ACC_survival.tsv')
tr.data_preprocess()
maven = ma.multi_ae(tr, 10, 20, 100, 500, 0.80)
maven.tf_construct(tr)
maven.pre_train()
maven.data_train()


