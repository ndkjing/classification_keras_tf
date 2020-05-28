import tensorflow as tf
from tensorflow.python.platform import gfile

# 最初pb模型文件路径
base_dir = 'C:\PythonProject\keras_to_tensorflow\\nas\\'

model_name = 'saved_model.pb'

sess = tf.Session()
with gfile.FastGFile(base_dir+model_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # graph = tf.get_default_graph()
    # tf.import_graph_def(graph_def, name='graph')
    # summaryWriter = tf.summary.FileWriter('log/', graph)
    # print(graph_def)
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

# saved_model_dir = 'model_1'
# sess.graph.as_default()
# builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
# builder.add_meta_graph_and_variables(sess, ['tag_string'])
# builder.save()
# meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], saved_model_dir)
# TODO
x = sess.graph.get_tensor_by_name('input_1:0')
y = sess.graph.get_tensor_by_name('3333/Softmax:0')

print(x,y)
# 目标路径
saved_model_dir = 'nas'
builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
# x 为输入tensor, keep_prob为dropout的prob tensor
inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x)}

# y 为最终需要的输出结果tensor
outputs = {'output' : tf.saved_model.utils.build_tensor_info(y)}

signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')

builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature':signature})
builder.save()









