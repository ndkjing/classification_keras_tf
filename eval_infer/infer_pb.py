import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import tensorflow as tf
import numpy as np
print(tf.__version__)


loaded = tf.saved_model.load('/home/create/jing/tfmodels/models/research/object_detection/ssd_model/saved_model')
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

labeling = infer(tf.constant(np.array(np.random.random((1,300,300,3)),dtype=np.uint8)))

print(labeling)