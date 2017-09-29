import train_set as ts
import c2v_models as cv
import tensorflow as tf
import model_builder as mb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tr = ts.data_set("ACC", 45, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()
tr.data_rearrange()
maven = cv.FarSeer(tr, 0.888)

maven.SurvivalNet(3000)
#mb.create_model(True)




