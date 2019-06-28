#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys 
import time
#reload(sys)
#sys.setdefaultencoding('utf-8')
import tensorflow as tf
from DataRead import DataReadAndNegSamp, getNow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

dataClass  = DataReadAndNegSamp(file_input='../data/20190623.ID')
train_data = dataClass.train_data
hidden_dim = 128 
batch_size = 1024*48
epoch_num  = 8 
row_num, col_num = train_data.shape
col_max    = train_data.max()
max_index  = col_max.max()
print ('max_index:', max_index)

with tf.device('/gpu:1'):
  with tf.variable_scope('weight'):
    w = tf.get_variable(name = 'w', dtype = tf.float32, trainable=True, shape = [max_index+1, hidden_dim], \
        regularizer = tf.contrib.layers.l2_regularizer(0.01), \
        initializer = tf.contrib.layers.xavier_initializer())
  with tf.variable_scope('compute'):
    x = tf.placeholder(name='x', shape=[None, 13], dtype=tf.int32)
    y = tf.placeholder(name='y', shape=[None, 1], dtype=tf.int32)
    xx= tf.nn.embedding_lookup(w, x)
    user, histURL, targetURL = tf.split(xx, num_or_size_splits=[1, 11, 1], axis=1)
    sim  = tf.matmul(targetURL, tf.transpose(histURL, [0, 2, 1]))
    sim  = tf.reduce_mean(sim , axis=1)
    user = tf.reduce_mean(user, axis=1)
    targetURL = tf.reduce_mean(targetURL, axis=1)
    
    flatten_layer = tf.concat([user, sim, targetURL], axis=1)
    print ('debug, x.shape :', x.get_shape())
    print ('debug, user.shape:', user.get_shape())
    print ('debug, histurls.shape:', histURL.get_shape())
    print ('debug, targeturls.shape:', targetURL.get_shape())
    print ('debug, flatten_layer.shape:', flatten_layer.get_shape())
    hidden_layer1 = tf.layers.dense(inputs= flatten_layer, units= 500, activation= 'relu', \
                    use_bias= True, kernel_initializer= tf.contrib.layers.xavier_initializer())
    hidden_layer2 = tf.layers.dense(inputs= hidden_layer1, units= 200, activation= 'relu', \
                    use_bias= True, kernel_initializer= tf.contrib.layers.xavier_initializer())
    output = tf.layers.dense(inputs= hidden_layer2, units=1, activation= None, use_bias=True)
    logits = output
    prob   = tf.nn.sigmoid(logits)
    plabel = tf.cast((prob>0.5), tf.float32)
    labels = tf.cast(y, tf.float32)
    loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    accuracy = tf.metrics.accuracy(labels, plabel) ## (acc, acc_op) 
    #correct_prediction = tf.equal(labels, plabel)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    train_op  = optimizer.minimize(loss = loss, var_list=train_var)

gpu_options = tf.GPUOptions(allow_growth = True)
config=tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  #sess.graph.finalize()
  for num in range(epoch_num):
    print (getNow(), 'now shuffle begin')
    train_data = dataClass.samplingagain()
    x_data = train_data[dataClass.col_names]
    y_data = train_data[['label']]
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.0005)
    print ('alldata.shape:', train_data.shape)
    print ('train_x.shape:', train_x.shape, 'train_y.shape:', train_y.shape)
    print ('test _x.shape:', test_x.shape,  'test _y.shape:', test_y.shape)
    print (getNow(), 'now shuffle and split end')
    for i in range(int(train_x.shape[0]/batch_size)):
      input_x = train_x[i*batch_size:(i+1)*batch_size]
      input_y = train_y[i*batch_size:(i+1)*batch_size]
      sess.run(train_op, feed_dict={x:input_x ,y:input_y})
      if i % 100 == 0:
        train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x:input_x, y:input_y})
        test_accuracy , test_loss  = sess.run([accuracy, loss], feed_dict={x:test_x, y:test_y})
        print (getNow(), 'iter:', num, 'setp:', i, 'train_accuracy:', train_accuracy[0], 'test_accuracy:', test_accuracy[0], 'train_loss:', train_loss, 'test_loss:', test_loss)

  ## now export model ##
  export_path_base = '../model/'
  model_version = time.strftime("%Y%m%d%H%M", time.localtime(float(time.time())))
  export_path = os.path.join( \
                tf.compat.as_bytes(export_path_base), \
                tf.compat.as_bytes(str(model_version)))
  print (getNow(), 'will export trained model to :', export_path_base)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_output_prob = tf.saved_model.utils.build_tensor_info(prob)
  tensor_info_output_label= tf.saved_model.utils.build_tensor_info(plabel)

  pred_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs = {'input' : tensor_info_input},
      outputs= {'predict_prob': tensor_info_output_prob, 'predict_label': tensor_info_output_label},
      method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
  ))
  '''
  pred_class_signature = ( 
    tf.saved_model.signature_def_utils.build_signature_def( 
      inputs = {'input_' : tensor_info_input},
      outputs= {'output_': tensor_info_output_label},
      method_name = tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
  ))
  '''
  builder.add_meta_graph_and_variables(
    sess = sess, tags = [tf.saved_model.tag_constants.SERVING], # tags='serve' 
    signature_def_map = {'predict' : pred_signature,
                         'serving_default': pred_signature},
    #main_op = tf.tables_initializer(),
    strip_default_attrs = True,
    clear_devices = True)

  builder.save()
  print (getNow(), 'export done')
  print (getNow(), 'now train end')
