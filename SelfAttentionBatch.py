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

def SelfAttentionBatchMultiHeads(query_layer, key_layer, value_layer):
  attend_z  = tf.matmul(query_layer, tf.transpose(key_layer, [0, 1, 3, 2]))
  size_perH = tf.cast(query_layer.get_shape().as_list()[-1], tf.float32)
  scaled_z  = tf.multiply(attend_z, 1/tf.sqrt(size_perH))
  softmax_z = tf.nn.softmax(scaled_z, axis=3) # equal dim=-1 # or default #
  multi_att = tf.matmul(softmax_z, value_layer)
  return multi_att

def MultiHeadsDenseLayer(tensor_input, w, name='dense_layer', hidden_size=None, heads_num=2):
  #batch, seq_length, emb_size = tensor_input.get_shape().as_list()
  _, seq_length, emb_size = tensor_input.get_shape().as_list()
  if hidden_size: ## it means specify heads_num*size_per_head ##
    size_per_head = int(hidden_size/heads_num)
    print 'hidden_size is specify, size_per_head:', size_per_head, 'heads_num:', heads_num, 'hidden_size:', hidden_size
  else: ## it means multi-heads all-size is equal to input-size ##
    hidden_size   = emb_size
    size_per_head = int(hidden_size/heads_num)
    print 'hidden_size not specify, size_per_head:', size_per_head, 'heads_num:', heads_num, 'hidden_size:', hidden_size

  tensor_input_2d    = tf.reshape(tensor_input, [-1, emb_size])
  tensor_input_layer = tf.matmul(tensor_input_2d, w)
  tensor_input_layer = tf.reshape(tensor_input_layer, [-1, seq_length, heads_num, size_per_head])
  tensor_input_layer = tf.transpose(tensor_input_layer, [0, 2, 1, 3]) 
  return tensor_input_layer 


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
