#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
  use multi-gpu in one model
  in the way of split data-array
  version-1: 验证了clip_norm的正确性 、net_func在外侧的可行性、reg_loss的隐式计算
'''
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf

def build_column():
  feat1_column = tf.feature_column.numeric_column('feat1')
  feat2_column = tf.feature_column.numeric_column('feat2')
  feat3_column = tf.feature_column.numeric_column('feat3')
  feat4_column = tf.feature_column.numeric_column('feat4')
  feature_columns = [feat1_column, feat2_column, feat3_column, feat4_column]
  return feature_columns

iris_data_file = './iris.data'
def input_fn(file_name, epoch= 10, shuffle=False, batch_size=10):
  columns = build_column()
  def decode_line(line):
    columns = tf.decode_csv(line, \
            record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])
    #return columns[:-1], columns[-1]
    return dict(zip(['feat1', 'feat2', 'feat3', 'feat4', 'label'], columns))
  #def tensor_as_input(input, columns):
  #  return tf.feature_column.input_layer(features = input, feature_columns=columns, trainable=True)
  dataset = tf.contrib.data.TextLineDataset(file_name)
  dataset = dataset.map(decode_line, num_threads = 100)
  if shuffle: dataset = dataset.shuffle(buffer_size = 100)
  dataset = dataset.repeat(count = epoch)
  dataset = dataset.batch(batch_size = batch_size)
  dataset = dataset.make_one_shot_iterator()
  features = dataset.get_next()
  y  = features.pop('label')
  y  = tf.one_hot(y, 3)
  #print '#debug, x:', features, 'y:', y
  return features, y
  #return tensor_as_input(features, columns), y
def net_func(x):
  # linear model #
  # regularizer will be collected into GraphKeys #
  with tf.variable_scope('layer-1'):
    w1 = tf.get_variable('w1', initializer=tf.random_normal(shape=[4,8], mean=0.0, stddev=1.0), \
                         dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b1 = tf.get_variable('b1', initializer=tf.zeros(shape=[1,8], dtype=tf.float32))
  with tf.variable_scope('layer-2'):
    w2 = tf.get_variable('w2', initializer=tf.random_normal(shape=[8,3], mean=0.0, stddev=1.0), \
                         dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b2 = tf.get_variable('b2', initializer=tf.zeros(shape=[1,3], dtype=tf.float32))
  with tf.variable_scope('compute'):
    y1 = tf.matmul(x, w1) + b1
    y2 = tf.matmul(y1,w2) + b2
  return y2
  
def mode_mine(features, labels, mode, params):
  x  = tf.feature_column.input_layer(features=features, feature_columns=params['columns'])
  logits  = net_func(x)
  prob        = tf.nn.softmax(logits, dim=1)
  prob_class  = tf.argmax(prob, axis=1)
  ## predict mode ##
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions_op = {'prob': prob, 'prob_class': prob_class}
    return tf.estimator.EstimatorSpec(mode, predictions=predictions_op)
  ## train mode ##
  # regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) ## 
  # 这个时候，不能显示地添加正则，但是variable_scope或者tf.get_variable会隐藏着regular形参, #
  # 可以通过tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)获取到 #
  #print '#debug', tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  #print '#debug', tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  #print '#debug', tf.losses.get_regularization_loss()
  #reg_loss    = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  reg_loss    = tf.losses.get_regularization_loss(scope=None)
  loss        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits=logits)) + reg_loss
  optimizer   = tf.train.AdamOptimizer(learning_rate=0.001)
  gradients, variables = zip(*optimizer.compute_gradients(loss))
  gradients, _= tf.clip_by_global_norm(t_list=gradients, clip_norm=100.0) ## will slow down the procesee appearently ##
  # t_list[i] * clip_norm / max(global_norm, clip_norm)
  # where, global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
  # if clip_norm > global_norm, then return t_list itself
  train_op    = optimizer.apply_gradients(grads_and_vars = zip(gradients, variables), global_step = tf.train.get_global_step())
  #train_op    = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  if mode == tf.estimator.ModeKeys.TRAIN:
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  ## eval mode ##
  accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=prob_class)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy':accuracy})

def main(unused_argv):
  run_config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto( \
                                                                    gpu_options          =tf.GPUOptions(allow_growth=True), \
                                                                    device_count         = {'GPU':0}, \
                                                                    allow_soft_placement = True))
  params= {}; params['GPU'] = 4; params['columns']=build_column()
  model = tf.estimator.Estimator( \
             model_fn = mode_mine, \
             params   = params, \
             config   = run_config, \
             model_dir= './model')
  for n in range(100//1):
    model.train(input_fn = lambda: input_fn('./iris.data', 1, True, 16))
    eval_res = model.evaluate(input_fn = lambda: input_fn('./iris.data', 1, False, 16))
    print 'epoch:', (n+1)*1, time.ctime()
    print '-'*40
    for key in sorted(eval_res):
      print key, eval_res[key]
if __name__ =='__main__':
  tf.app.run(main=main, argv=[sys.argv[0]]) 
