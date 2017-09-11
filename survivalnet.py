import train_set as ts
import tensorflow as tf

tr = ts.data_set("ACC", 45, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()
tr.data_rearrange()


