#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

def SelfAttentionBatch(query, key, value):
  ''' 
    input-query: [batch, seq_len1, hidden_dim]
    input-key  : [batch, seq_len2, hidden_dim]
    input-value: [batch, seq_len2, hidden_dim]
    output : [batch, seq_len1, hidden_dim]
  '''
  sim_comp = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
  scaled_v = tf.multiply(sim_comp, 1/tf.sqrt(4.0))
  softmax_ = tf.nn.softmax(scaled_v, axis=2)
  output   = tf.matmul(softmax_, value)
  return output
  

if __name__=='__main__':
  query = tf.Variable(tf.constant( \
      [[[1.1, 4.2, 1.3, 1.4], \
       [0.1, 2.3, 2.5, 2.7], \
       [3.1, 3.4, 3.8, 3.0]],\
  
      [[1.1, 1.2, 1.3, 1.4], \
       [2.1, 2.2, 2.3, 2.4], \
       [3.1, 3.2, 3.3, 3.4]],\
  
      [[0.1, 0.2, 0.3, 0.4], \
       [1.1, 1.2, 2.3, 2.4], \
       [0.1, 0.2, 3.3, 3.4]]], \
      dtype=tf.float32), name='query')
  ## if query shape : [batch, 1, emb_size] 
  query = tf.Variable(tf.constant( \
        [[[1.1, 4.2, 1.3, 1.4]], \
         [[1.1, 1.2, 1.3, 1.4]], \
         [[0.1, 0.2, 0.3, 0.4]]], \
        dtype=tf.float32), name ='query')
  
  
  key   = tf.Variable(tf.constant( \
      [[[1.1, 4.2, 1.3, 1.4], \
       [0.1, 2.3, 2.5, 2.7], \
       [3.1, 3.4, 3.8, 3.0]],\
  
      [[1.1, 1.2, 1.3, 1.4], \
       [2.1, 2.2, 2.3, 2.4], \
       [3.1, 3.2, 3.3, 3.4]],\
  
      [[0.1, 0.2, 0.3, 0.4], \
       [1.1, 1.2, 2.3, 2.4], \
       [0.1, 0.2, 3.3, 3.4]]], \
      dtype=tf.float32), name='key')
  value = tf.Variable(tf.constant( \
      [[[1.1, 4.2, 1.3, 1.4], \
       [0.1, 2.3, 2.5, 2.7], \
       [3.1, 3.4, 3.8, 3.0]],\

      [[1.1, 1.2, 1.3, 1.4], \
       [2.1, 2.2, 2.3, 2.4], \
       [3.1, 3.2, 3.3, 3.4]],\

      [[0.1, 0.2, 0.3, 0.4], \
       [1.1, 1.2, 2.3, 2.4], \
       [0.1, 0.2, 3.3, 3.4]]], \
      dtype=tf.float32), name='value')

  gpu_options = tf.GPUOptions(allow_growth = True)
  with tf.Session(config = tf.ConfigProto( \
                  gpu_options = gpu_options, \
                  allow_soft_placement = True, \
                  log_device_placement = False)) as sess:
    sess.run(tf.global_variables_initializer())
    attention_vec = SelfAttentionBatch(query, key, value)
    print 'input.query.shape:', sess.run(query).shape
    print 'input.key.shape  :', sess.run(key).shape
    print 'input.value.shape:', sess.run(value).shape
    print 'output.shape     :', sess.run(attention_vec).shape
