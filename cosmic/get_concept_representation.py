import numpy as np
import tensorflow as tf

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

reader = tf.compat.v1.train.NewCheckpointReader('./model_100_without/model.ckpt-24')
dic = reader.get_variable_to_shape_map()
#print(dic)
w = reader.get_tensor("embedding/Adam")
print(w.shape)
data=standardization(w)
#for p in data:
    #print(p)
print(data.shape)
np.save('D:\\image_tag\\tag_dataset\\embed\\embedding_vec_100_new_without_text_v2.npy',data)