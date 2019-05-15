#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
from DataRead import DataReadAndNegSamp, getNow
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import shutil
tf.logging.set_verbosity(tf.logging.INFO) ## hook-logging alos print ##
 
train_data = DataReadAndNegSamp(file_input='./user_click_urls.ID.small').train_data
hidden_dim = 128 
batch_size = 1024*32 
epoch_num  = 10
row_num, col_num = train_data.shape
col_max    = train_data.max()
max_index  = col_max.max()
x_data = train_data[['user', 'url', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']]
y_data = train_data[['label']]
train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.1)
def get_input_from_pd(x, y, num_epochs=1, shuffle=True, batch_size=batch_size):
  return tf.estimator.inputs.pandas_input_fn(
    x = x, \
    y = y, \
    num_epochs = num_epochs, \
    shuffle = shuffle, \
    batch_size = batch_size, \
    num_threads = 2 
    )   

def build_feature_columns():
  user_id = tf.feature_column.numeric_column('user')
  url_id  = tf.feature_column.numeric_column('url')
  tag_id1 = tf.feature_column.numeric_column('tag1')
  tag_id2 = tf.feature_column.numeric_column('tag2')
  tag_id3 = tf.feature_column.numeric_column('tag3')
  tag_id4 = tf.feature_column.numeric_column('tag4')
  tag_id5 = tf.feature_column.numeric_column('tag5')
  return [user_id, url_id, tag_id1, tag_id2, tag_id3, tag_id4, tag_id5]

def model_mine(features, labels, mode, params):
  with tf.device('/gpu:0'): ## 若变量很大,会在GPU上分配不足 ##
    with tf.variable_scope('weight'):
#      w = tf.get_variable(name = 'w', dtype = tf.float32, trainable=True, \
#          regularizer = tf.contrib.layers.l2_regularizer(0.1), \
#          initializer = tf.random_normal(shape=[max_index+1,hidden_dim], mean=0.0, stddev=1.0))
      w = tf.get_variable(name = 'w', dtype = tf.float32, trainable=True, shape = [max_index+1, hidden_dim], \
          regularizer = tf.contrib.layers.l2_regularizer(0.1), \
          initializer = tf.contrib.layers.xavier_initializer())
    print 'debug table w.shape:', w.get_shape(), 'max_index:', max_index

  x = tf.feature_column.input_layer(features, params['feature_columns'])
  x = tf.cast(x, tf.int32)
  x = tf.nn.embedding_lookup(w, x)
  x1, x2 = tf.split(x, num_or_size_splits=[1, 6], axis=1)
  user_input = tf.reduce_mean(x1, axis=1)
  urls_input = tf.reduce_mean(x2, axis=1)
  global_step= tf.train.get_or_create_global_step()
  #print 'user_input.shape:', user_input.get_shape()
  #print 'urls_input.shape:', urls_input.get_shape()
  #print 'matmul.shape:', tf.matmul(user_input, urls_input, transpose_b=True).get_shape()
  logits = tf.reduce_mean(tf.matmul(user_input, urls_input, transpose_b=True), axis=1)
  prob   = tf.sigmoid(logits)
  plabel = tf.cast((prob>0.5), tf.float32)
  labels = tf.cast(labels, tf.float32)
  loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits))
  #accuracy = tf.contrib.metrics.accuracy(labels = labels, predictions = plabel, name ='accuracy')
  accuracy = tf.metrics.accuracy(labels = labels, predictions = plabel, name ='accuracy')
  tf.summary.scalar('accuracy', accuracy[0])
  tf.summary.scalar('loss', loss)
  optimizer= tf.train.AdamOptimizer(learning_rate = 0.005)
  train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())

  predictions = {'prob':prob, 'plabel':plabel, 'accuracy':accuracy}
  #print 'all-keys:', tf.Graph.get_all_collection_keys()
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode, predictions = predictions)
  logging_hook = tf.train.LoggingTensorHook(tensors={'train-accuracy':accuracy[0], 'train-loss':loss}, every_n_iter=100)
  ## 这个钩子如果从模型外往里传，会报错变量accuracy不存在 ##
  if mode == tf.estimator.ModeKeys.TRAIN:
    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op, training_hooks=[logging_hook])
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = {'accuracy': predictions['accuracy']})
def build_estimator(model_dir):
  run_config = tf.estimator.RunConfig().replace( \
               session_config = tf.ConfigProto( \
                                gpu_options = tf.GPUOptions(allow_growth=True), \
                                allow_soft_placement = True))
  params = {}
  params['batch_size'] = batch_size
  params['feature_columns'] = build_feature_columns()
  return tf.estimator.Estimator( \
         model_fn  = model_mine, \
         model_dir = model_dir, \
         config    = run_config, \
         params    = params)

model_dir = '../model/'
shutil.rmtree(model_dir)
os.mkdir(model_dir)
model = build_estimator(model_dir)
train_input_fn = get_input_from_pd(train_x, train_y, shuffle=True)
test_input_fn  = get_input_from_pd(test_x, test_y, shuffle=False)
print 'debug, train_x.shape:', train_x.shape, 'train_y.shape:', train_y.shape
print 'debug, test_x.shape :', test_x.shape,  'test_y.shape :', test_y.shape
epoch_per_eval = 1

for n in range(epoch_num):
  model.train(input_fn = train_input_fn, hooks=[])
  res = model.evaluate(input_fn = test_input_fn)
  print getNow(), 'epoch:', n, 'step:', global_step, 'test accuracy:', res['accuracy']
  #for key in sorted(res): print key, res[key]

'''
notice:
  1) hand code is faster than tf.estimator
  2) batch_size need to set large enough is epanded GPU power
  3) neg sampling need again to do each epoch, especial hand neg sampling
  4) accuracy in tf.estimator could be tf.metrics.accuracy; bu in hand code as follow
     # correct_prediction = tf.equal( tf.argmax(y_, 1), tf.argmax(output, 1))
     # accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32))
  5) 
'''
