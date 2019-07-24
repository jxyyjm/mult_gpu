#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
  1) input is seq UserID:UrlIDs
     table-1: UserID: UserTagsID
     table-2: UrlID : UrlTagsID
  2) how to construct neg-samples
  3) lookup twice, first for TagsID, second for embs
  4) add target-url similary with history-urls as feature
  5) add target-url attention with history-urls as feature # drop this feature #
  6) add target-url multi-heads attention with interest as feature
'''

import os
import sys 
import time
#reload(sys)
#sys.setdefaultencoding('utf-8')
import numpy as np
import pandas as pd
import tensorflow as tf
#from DataRead import DataReadAndNegSamp, getNow
from DataRead2 import DataReadAndNegSamp, getNow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from SelfAttentionBatch import SelfAttentionBatch, SelfAttentionBatchMultiHeads, MultiHeadsDenseLayer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#dataClass  = DataReadAndNegSamp(file_input='../data/20190702.ID.clean')
#dataClass  = DataReadAndNegSamp(file_input='../data/20190716.ID.clean')
#dataClass  = DataReadAndNegSamp(file_input='../data/20190716.ID.clean', file_negative='../data/20190716.ID.neg_real')
dataClass  = DataReadAndNegSamp(file_input='../data/20190716.ID.clean', file_negative='../data/20190716.ID.neg_real.all')
train_data = dataClass.train_data
hidden_dim = 128 
hidden_size= 128*2
heads_num  = 2 
batch_size = 1024*16
epoch_num  = 15
seq_len    = 12
row_num, col_num = train_data.shape
col_max    = train_data[['user']].max()
max_id_user= col_max.max() # for userID == > userTagsID #
col_max    = train_data[['seq'+str(i) for i in range(seq_len)]].max()
max_id_urls= col_max.max() # for urlsID == > urlsTagsID #
print ('max_index, max_user:', max_id_user, 'max_urls:', max_id_urls)

col_names = ['user_tag_'+str(i) for i in range(20)]
user2tags = pd.read_csv('../data/key2id.20190716.user.tags.ID.only', sep='\t', names=col_names, header=None, index_col=0, dtype=np.int32)
maxID_tag1= user2tags.max().max() # for userTagsID == > tagsHidden #
col_names = ['url_tag_'+str(i) for i in range(7)]
urls2tags = pd.read_csv('../data/key2id.20190716.url.tags.ID.only', sep='\t', names=col_names, header=None, index_col=0, dtype=np.int32)
maxID_tag2= urls2tags.max().max() # for urlsTagsID == > tagsHidden #

with tf.device('/gpu:3'):
  with tf.variable_scope('weight'):
    ## table for ID ==> tagsID ##
    table_user2tags = tf.get_variable(name = 'table_user2tags', initializer = tf.constant(user2tags), dtype = tf.int32, trainable=False)
    table_urls2tags = tf.get_variable(name = 'table_urls2tags', initializer = tf.constant(urls2tags), dtype = tf.int32, trainable=False)
    ## table for tagsID ==> Hidden ##
    w_user = tf.get_variable(name = 'w_user', dtype = tf.float32, trainable=True, shape = [maxID_tag1+1, hidden_dim], \
                            regularizer = tf.contrib.layers.l2_regularizer(0.01), \
                            initializer = tf.contrib.layers.xavier_initializer())
    w_urls = tf.get_variable(name = 'w_urls', dtype = tf.float32, trainable=True, shape = [maxID_tag2+1, hidden_dim], \
                            regularizer = tf.contrib.layers.l2_regularizer(0.01), \
                            initializer = tf.contrib.layers.xavier_initializer())
    ## var for query/key/value ==> query_layer/key_layer/value_layer ##
    w_query= tf.get_variable(name = 'w_query', dtype = tf.float32, trainable = True, shape = [hidden_dim, hidden_size], \
                            regularizer = tf.contrib.layers.l2_regularizer(0.01), \
                            initializer = tf.contrib.layers.xavier_initializer())
    w_key  = tf.get_variable(name = 'w_key', dtype = tf.float32, trainable = True, shape = [hidden_dim, hidden_size], \
                            regularizer = tf.contrib.layers.l2_regularizer(0.01), \
                            initializer = tf.contrib.layers.xavier_initializer())
    w_value= tf.get_variable(name = 'w_value', dtype = tf.float32, trainable = True, shape = [hidden_dim, hidden_size], \
                            regularizer = tf.contrib.layers.l2_regularizer(0.01), \
                            initializer = tf.contrib.layers.xavier_initializer())

  with tf.variable_scope('compute'):
    x = tf.placeholder(name='x', shape=[None, 13], dtype=tf.int32)
    y = tf.placeholder(name='y', shape=[None, 1], dtype=tf.int32)
    ## here split ##
    userID, histURLID, targetURLID = tf.split(x, num_or_size_splits=[1, 11, 1], axis=1)
    ## here lookup ## notice ID ==> TagsID #
    userTagsID      = tf.nn.embedding_lookup(table_user2tags, userID)
    histURLTagsID   = tf.nn.embedding_lookup(table_urls2tags, histURLID)
    targetURLTagsID = tf.nn.embedding_lookup(table_urls2tags, targetURLID)
    print 'userTagsID.shape     :', userTagsID.get_shape()
    print 'urlsTagsID.shape hist:', histURLTagsID.get_shape()
    print 'urlsTagsID.shape targ:', targetURLTagsID.get_shape()
    ## input format for serving ##
    input_tensor = tf.concat([ \
                   tf.reshape(userTagsID,      [-1, 20])  , \
                   tf.reshape(histURLTagsID,   [-1, 11*7]), \
                   tf.reshape(targetURLTagsID, [-1, 7])  ], \
                   axis=1, name='input')
    ## here lookup ## notice TagsID ==> HiddenDim ## 
    user, hist, targ = tf.split(input_tensor, num_or_size_splits=[20, 11*7, 7], axis=1)
    user = tf.nn.embedding_lookup(w_user, user)
    hist = tf.nn.embedding_lookup(w_urls, hist)
    targ = tf.nn.embedding_lookup(w_urls, targ)
    print 'user.shape hidden:', user.get_shape()
    print 'hist.shape hidden:', hist.get_shape()
    print 'targ.shape hidden:', targ.get_shape()
    ## mean each url-filed user-filed ##
    user = tf.reshape(user, [-1, 1, 20, hidden_dim])
    hist = tf.reshape(hist, [-1, 11, 7, hidden_dim])
    targ = tf.reshape(targ, [-1, 1,  7, hidden_dim])
    print 'after reshape for sim-compute, user.shape:', user.get_shape()
    print 'after reshape for sim-compute, hist.shape:', hist.get_shape()
    print 'after reshape for sim-compute, targ.shape:', targ.get_shape()

    user = tf.reduce_mean(user, axis=2, keepdims=None)
    hist = tf.reduce_mean(hist, axis=2, keepdims=None)
    targ = tf.reduce_mean(targ, axis=2, keepdims=None)
    print 'user.shape after mean:', user.get_shape() # [batch, 1, 128] #
    print 'hist.shape after mean:', hist.get_shape() # [batch, 11,128] #
    print 'targ.shape after mean:', targ.get_shape() # [batch, 1, 128] #
    ## attention compute ##
    sim  = tf.matmul(targ, tf.transpose(hist, [0, 2, 1]))
    print 'sim.shape :', sim.get_shape()
    query= targ
    key  = tf.concat([user, hist], axis=1)
    value= key
    query_layer = MultiHeadsDenseLayer(query, w_query, name='query_layer', hidden_size=hidden_size, heads_num=heads_num)
    key_layer   = MultiHeadsDenseLayer(key,   w_key,   name='key_layer',   hidden_size=hidden_size, heads_num=heads_num)
    value_layer = MultiHeadsDenseLayer(value, w_value, name='value_layer', hidden_size=hidden_size, heads_num=heads_num)
    mh_atten_target_with_interest = SelfAttentionBatchMultiHeads(query_layer, key_layer, value_layer)
    #attention_target_with_hist = SelfAttentionBatch(query=targ, key=hist, value=hist)
    print 'query.shape:', query.get_shape(), '==> query_layer.shape:', query_layer.get_shape()
    print 'key.shape  :', key.get_shape(),   '==> key_layer.shape  :', key_layer.get_shape()
    print 'value.shape:', value.get_shape(), '==> value_layer.shape:', value_layer.get_shape()
    print 'atten.shape:', mh_atten_target_with_interest.get_shape()
    #print 'att.shape :', attention_target_with_hist.get_shape()
    #_, dim2, dim3 = hist.get_shape().as_list()
    user = tf.reshape(user, [-1, hidden_dim])
    sim  = tf.reshape(sim , [-1, 11])
    targ = tf.reshape(targ, [-1, hidden_dim])
    _, heads_num_here, seq_len_here, size_per_head_here = mh_atten_target_with_interest.get_shape().as_list()
    mh_atten_target_with_interest = tf.reshape(tf.transpose(mh_atten_target_with_interest, [0, 2, 1, 3]), [-1, seq_len_here, hidden_size])
    print 'after concate multi-heads, atten.shape :', mh_atten_target_with_interest.get_shape()
    attention_output = tf.reduce_mean(mh_atten_target_with_interest, axis=1, keepdims=None)
    #attention_output = tf.reshape(mh_atten_target_with_interest, [-1, seq_len_here*hidden_size]) # not a good method #
    #attention_output = tf.matmul(mh_atten_target_with_interest, w_attention_trans) # here is one another method #
    print 'after reshape for one-output, att.shape:', mh_atten_target_with_interest.get_shape()
    #attention_target_with_hist = tf.reshape(attention_target_with_hist, [-1, hidden_dim])
    print 'after reshape for hid-input, user.shape:', user.get_shape()
    print 'after reshape for hid-input, hist.shape:', hist.get_shape(), '[notice: not used after here]'
    print 'after reshape for hid-input,  sim.shape:',  sim.get_shape()
    print 'after reshape for hid-input, targ.shape:', targ.get_shape()
    print 'after reshape for hid-input, atte.shape:', attention_output.get_shape()
    #print 'after reshape for hid-input, atte.shape:', attention_target_with_hist.get_shape()

    #flatten_layer = tf.concat([user, sim, targ, attention_target_with_hist], axis=1)
    flatten_layer = tf.concat([user, sim, targ, attention_output], axis=1)
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

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
  #tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
  tensor_info_output_prob = tf.saved_model.utils.build_tensor_info(prob)
  tensor_info_output_label= tf.saved_model.utils.build_tensor_info(plabel)

  pred_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs = {'input' : tensor_info_input},
      outputs= {'predict_prob': tensor_info_output_prob, 'predict_label': tensor_info_output_label},
      method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
  ))
  ## here use multi output ##
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
