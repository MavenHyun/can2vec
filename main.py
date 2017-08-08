import multi_ae as ma
import split_ae as sa
import train_set as ts
import c2v_models as cv

tr = ts.data_set('ACC_features.tsv', 'ACC_survival.tsv', 50, 80)
tr.data_extract()
tr.data_preprocess()


maven = cv.SeerAdept(tr, 45, 30)
ae = maven.training_autoencoder('cli', 19, 'relu')
maven.mirror_image(ae, maven.P['cli'], 'adam', 50001, 0.0001)

maven = cv.SeerAdept(tr, 45, 30)
ae = maven.training_autoencoder('mut', 101, 'relu')
maven.mirror_image(ae, maven.P['mut'], 'adam', 50001, 0.0001)

maven = cv.SeerAdept(tr, 45, 30)
ae = maven.training_autoencoder('CNV', 800, 'relu')
maven.mirror_image(ae, maven.P['CNV'], 'adam', 50001, 0.0001)

maven = cv.SeerAdept(tr, 45, 30)
ae = maven.training_autoencoder('mRNA', 10000, 'relu')
maven.mirror_image(ae, maven.P['mRNA'], 'adam', 50001, 0.0001)



