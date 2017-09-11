import train_set as ts
import c2v_models as cv
import tensorflow as tf
import model_builder as mb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mb.create_model(True)




