import os
import gc
import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
from DataRead import DataRead, getNow
from DataReadRaw import DataReadRaw
from attention import fcn_self_attention
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
     ==> Could I define two input, YES.
     ID-input used when quick train, Raw-input used as tf-serving
  1) input : label + user + [[item,rytpe]] ==> 14+1+4*14=71
  2) layer1: network is constructed using single-attention
  3) layer2: cross-dot as this layer [simple to convergence]
  4) mask  : the click-pos and random negative-pos will mask
  5) saveModel: use raw data as input,  
  notice: log is 2019121011.log
  notice: this saver could be optimal by freezen_graph[here not used]
          ref: https://blog.eson.org/pub/3da24a26/

'''
hidden_dim = 32
#batch_size = 1024*32
batch_size = 256
epoch_num  = 25
attention_size = 32
attention_head = 2
seq_len = 14
filed_len = 4
user_len = 1
#dataClass  = DataRead(file_input='../data/seq.train.data.20191215.UserProcess.NoPos')
#testClass  = DataRead(file_input='../data/seq.train.data.20191215.UserProcess.NoPos.part')
dataClass  = DataRead( \
             #file_input = '../data/seq.train.data.20191215.UserProcess.NoPos.Raw.ID', \
             file_input = '../data/seq.test.data.20191216.UserProcess.NoPos.Raw.part.ID', \
             seq_len    = seq_len, \
             feat_len   = user_len+seq_len*filed_len)

testClass  = DataRead( \
             file_input = '../data/seq.test.data.20191216.UserProcess.NoPos.Raw.Clean.part.ID', \
             seq_len    = seq_len, \
             feat_len   = user_len+seq_len*filed_len)
testClassRaw = DataReadRaw( \
             file_input = '../data/seq.test.data.20191216.UserProcess.NoPos.Raw.Clean.part', \
             seq_len    = seq_len, \
             feat_len   = user_len+seq_len*filed_len)

train_data = dataClass.shuffle()
col_max    = train_data.max()
max_index  = col_max.max()
#max_index = 5428559
print 'max_index=', max_index

num_oov_buckets = 1
with tf.device('/gpu:1'):
  with tf.variable_scope('weight'):
    w = tf.get_variable(name = 'w', dtype = tf.float32, trainable=True, shape = [max_index+1+1, hidden_dim], \
        regularizer = tf.contrib.layers.l2_regularizer(0.002), \
        initializer = tf.contrib.layers.xavier_initializer())
    # the extended 1 is used as unknow ## notice: need to match following table-initialization #
    table = tf.contrib.lookup.index_table_from_file( \
            vocabulary_file = '/models/seq_model/used.kv', \
            default_value   = max_index+1, \
            #num_oov_buckets = num_oov_buckets, \
            key_column_index   = 0, \
            value_column_index = 1, \
            delimiter          = '\t')
    w_output = tf.get_variable(name='w_output', dtype=tf.float32, shape=[attention_size*seq_len, seq_len], \
        regularizer = tf.contrib.layers.l2_regularizer(0.002), \
        initializer = tf.contrib.layers.xavier_initializer())
    b_output = tf.get_variable(name='b_output', dtype=tf.float32, shape=[seq_len, 1],\
        initializer = tf.contrib.layers.xavier_initializer())


  with tf.variable_scope('compute'):
    y = tf.placeholder(name='y', shape=[None, seq_len], dtype=tf.int32)   # [batch, seq] #
    x = tf.placeholder(name='x', shape=[None, 1+seq_len*filed_len], dtype=tf.string) # [batch, 1+seqxfiled] #
    x_ids = table.lookup(x, name='x_ids')
    #x = tf.placeholder(name='x', shape=[None, 1+seq_len*filed_len], dtype=tf.int32) # [batch, seqxfiled] #
    print 'debug, x_ids.shape=', x_ids.get_shape()
    user, seq = tf.split(x_ids, num_or_size_splits=[1,seq_len*filed_len], axis=1)
    #user = [batch, 2] # seq = [batch, seq*filed]
    seq  = tf.reshape(seq, [-1, seq_len, filed_len]) #seq  = [batch, seq, filed ]
    print 'debug, seq.shape=', seq.get_shape(), 'user.shape=', user.get_shape()
    seqX = tf.nn.embedding_lookup(w, seq) # [index, dim].lookup [batch, seq, filed] = [batch, seq, filed, dim] #
    userX= tf.nn.embedding_lookup(w, user)# [bath, 2, dim]
    print 'debug, after emb.lookup, seqX.shape=', seqX.get_shape(), 'userX.shape=', userX.get_shape()
    seqX = tf.reduce_mean(seqX, axis=2, keepdims=False)  # ==> [batch, seq, dim] #
    userX = tf.reduce_mean(userX, axis=1, keepdims=True) # ==> [batch, 1,   dim] #
    print 'debug, after pooling, seqX.shape=', seqX.get_shape(), 'userX.shape=', userX.get_shape()
    att_output = fcn_self_attention(query=userX, key=seqX, value=seqX, attention_size=attention_size, heads=attention_head)
    print 'debug, att_output.shape=', att_output.get_shape() # (?, 2, 1, 16) = [batch, heads, seq, att_size/heads]
    _, heads_num_here, seq_len_here, size_per_head_here = att_output.get_shape().as_list()
    ## (?, 2, 1, 16) --> (?, 1, 2, 16) --> (?, 1, 2*16) # [batch, seq, att_size] #
    att_output = tf.reshape(tf.transpose(att_output, [0, 2, 1, 3]), [-1, seq_len_here, heads_num_here*size_per_head_here])
    print 'debug, att_output.reshape=', att_output.get_shape()
    #att_output = tf.reshape(att_output, [-1, heads_num_here*seq_len_here*size_per_head_here])
    #print 'debug, att_output.reshape=', att_output.get_shape()
    logits_ = tf.matmul(seqX, tf.transpose(att_output, [0, 2, 1]))
    print 'debug, logtis_.shape=', logits_.get_shape()
    logits = logits_ + b_output
    logits = tf.reshape(logits, (-1,seq_len))
    print 'debug, logits.shape=', logits.get_shape()
    prob   = tf.nn.sigmoid(logits)
    print 'debug, prob.shape=', prob.get_shape()
    plabel = tf.cast((prob>0.5), tf.float32)
    labels = tf.cast(y, tf.float32)
    #loss_  = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    ## here mask start ##
    pos_mask = tf.cast(tf.equal(y, 1), tf.int32)
    part_1   = tf.random_uniform(shape=[batch_size, seq_len],minval=0,maxval=2, dtype=tf.int32)
    part_2   = tf.random_uniform(shape=[batch_size, seq_len],minval=0,maxval=2, dtype=tf.int32)
    part_3   = tf.random_uniform(shape=[batch_size, seq_len],minval=0,maxval=2, dtype=tf.int32)
    neg_mask = tf.cast(tf.equal(tf.multiply(part_1, tf.multiply(part_2, part_3)), 1), tf.int32)
    all_mask = tf.bitwise.bitwise_or(pos_mask, neg_mask)
    labels_mask = tf.boolean_mask(labels, all_mask)
    logits_mask = tf.boolean_mask(logits, all_mask)
    loss_  = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_mask, logits=logits_mask)
    loss_a = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    ## here end mask ##
    loss   = tf.reduce_mean(loss_a)
    p_pos  = tf.cast(tf.equal(labels, 1), tf.float32)
    n_pos  = tf.cast(tf.equal(labels, 0), tf.float32)
    ploss  = tf.reduce_mean(tf.multiply(loss_a, p_pos))
    nloss  = tf.reduce_mean(tf.multiply(loss_a, n_pos))
    accuracy = tf.metrics.accuracy(labels, plabel) ## (acc, acc_op) 
    ## positive accuracy ##
    positive_mask = tf.equal(labels, 1)
    positive_accu = tf.metrics.accuracy( \
                    tf.boolean_mask(labels, positive_mask), \
                    tf.boolean_mask(plabel, positive_mask))
    ## negative accuracy ##
    negative_mask = tf.equal(labels, 0)
    negative_accu = tf.metrics.accuracy( \
                    tf.boolean_mask(labels, negative_mask), \
                    tf.boolean_mask(plabel, negative_mask))
    #all_auc   = tf.metrics.auc(tf.reshape(labels, (1, -1)), tf.reshape(prob, (1, -1)))
    all_auc   = tf.contrib.metrics.streaming_auc(labels=tf.reshape(labels, (1, -1)), predictions=tf.reshape(prob, (1, -1)))
    ## index-0 accuracy ##
    #first_accuracy= tf.metrics.accuracy(labels[:,0], plabel[:,0])
    first_accuracy= tf.metrics.accuracy(labels[:,0], plabel[:,0])
    first_auc = tf.contrib.metrics.streaming_auc(labels=labels[:,0], predictions=prob[:,0])
    #correct_prediction = tf.equal(labels, plabel)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
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
    print getNow(), 'now shuffle begin'
    train_data = dataClass.shuffle()
    x_data = train_data[['feat_'+str(i) for i in range(1+seq_len*filed_len)]]
    y_data = train_data[['label_'+str(i) for i in range(seq_len)]]
    #train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.0005)
    train_x, train_y = x_data, y_data
    test_data = testClass.data_read
    test_x = test_data[['feat_'+str(i) for i in range(1+seq_len*filed_len)]]
    test_y = test_data[['label_'+str(i) for i in range(seq_len)]]
    print 'alldata.shape:', train_data.shape
    print 'train_x.shape:', train_x.shape, 'train_y.shape:', train_y.shape
    print 'test _x.shape:', test_x.shape,  'test _y.shape:', test_y.shape
    test_data_raw = testClassRaw.data_read
    test_x_raw = test_data_raw[['feat_'+str(i) for i in range(1+seq_len*filed_len)]]
    test_y_raw = test_data_raw[['label_'+str(i) for i in range(seq_len)]]
    print 'test _x_raw.shape:', test_x.shape,  'test _y_raw.shape:', test_y.shape

    print getNow(), 'now shuffle and split end'
    for i in range(train_x.shape[0]/batch_size):
      ## 1) ori-input sequence data ##
      input_x = train_x[i*batch_size:(i+1)*batch_size]
      input_y = train_y[i*batch_size:(i+1)*batch_size]
      if input_x.shape[0] != batch_size: continue
      #sess.run(train_op, feed_dict={x:input_x, y:input_y})
      sess.run(train_op, feed_dict={x_ids:input_x, y:input_y})
      if i % 100 == 0:
        train_accuracy, train_loss, train_ploss, p_accu, n_accu, f_accu, a_auc, f_auc = sess.run([accuracy, loss, ploss, positive_accu, negative_accu, first_accuracy, all_auc, first_auc], feed_dict={x_ids:input_x, y:input_y})
        print getNow(), 'iter:', num, 'setp:', i, 'train accuracy:', train_accuracy[0], 'loss', train_loss, 'ploss', train_ploss, 'p_accu', p_accu[0], 'n_accu', n_accu[0], 'f_accu', f_accu[0], 'f_auc', f_auc[0], 'a_auc', a_auc[0]

        test_accuracy , test_loss, test_ploss, p_accu, n_accu, f_accu, a_auc, f_auc = sess.run([accuracy, loss, ploss, positive_accu, negative_accu, first_accuracy, all_auc, first_auc], feed_dict={x_ids:test_x, y:test_y})
        print getNow(), 'iter:', num, 'setp:', i, 'test  accuracy:', test_accuracy[0], 'loss', test_loss, 'ploss', test_ploss, 'p_accu', p_accu[0], 'n_accu', n_accu[0], 'f_accu', f_accu[0], 'f_auc', f_auc[0], 'a_auc', a_auc[0]

        test_accuracy , test_loss, test_ploss, p_accu, n_accu, f_accu, a_auc, f_auc = sess.run([accuracy, loss, ploss, positive_accu, negative_accu, first_accuracy, all_auc, first_auc], feed_dict={x:test_x_raw, y:test_y_raw})
        print getNow(), 'iter:', num, 'setp:', i, 'test  accuracy:', test_accuracy[0], 'loss', test_loss, 'ploss', test_ploss, 'p_accu', p_accu[0], 'n_accu', n_accu[0], 'f_accu', f_accu[0], 'f_auc', f_auc[0], 'a_auc', a_auc[0]

    ## memory clear hand ##
    try:
      if input_x.shape[0]>0:
        del input_x
        del input_y
        gc.collect()
    except: pass
    try:
      if new_input_x.shape[0]>0:
        del new_input_x
        del new_input_y
        gc.collect()
    except: pass
    if train_data.shape[0]>0:
      del train_data
      gc.collect()
    if x_data.shape[0]>0:
      del x_data
      del y_data
      gc.collect()
    if train_x.shape[0]>0:
      del train_x
      del train_y
      del test_x
      del test_y
      gc.collect()
  ## export model ##
  export_path_base = '/models/seq_model/'
  model_version = time.strftime('%Y%m%d%H%M', time.localtime(float(time.time())))
  export_path = os.path.join( \
                tf.compat.as_bytes(export_path_base), \
                tf.compat.as_bytes(str(model_version)))
  print getNow(), 'will export trained model into dir:', export_path_base
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_input_id = tf.saved_model.utils.build_tensor_info(x_ids)
  tensor_info_output_prob = tf.saved_model.utils.build_tensor_info(prob)      ## has bias, and use ##
  tensor_info_output_label= tf.saved_model.utils.build_tensor_info(plabel)    ## has bias, and use ##
  tensor_info_output_logits = tf.saved_model.utils.build_tensor_info(logits)  ## has bias, and use ##
  tensor_info_output_logits_= tf.saved_model.utils.build_tensor_info(logits_) ## has bias, not use ##

  pred_signature = ( \
                   tf.saved_model.signature_def_utils.build_signature_def( \
                   #inputs = {'input': tensor_info_input, 'input_id': tensor_info_input_id}, \
                   # if here used two as input, request will need two input #
                   inputs = {'input': tensor_info_input}, \
                   outputs= {'predict_prob': tensor_info_output_prob, \
                             'predict_label': tensor_info_output_label, \
                             'predict_logits': tensor_info_output_logits, \
                             'predict_logits_': tensor_info_output_logits_}, \
                   method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
  pred_signature_id = ( \
                   tf.saved_model.signature_def_utils.build_signature_def( \
                   inputs = {'input_id': tensor_info_input_id}, \
                   outputs= {'predict_prob': tensor_info_output_prob, \
                             'predict_label': tensor_info_output_label, \
                             'predict_logits': tensor_info_output_logits, \
                             'predict_logits_': tensor_info_output_logits_}, \
                   method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
  builder.add_meta_graph_and_variables( \
          sess = sess, tags = [tf.saved_model.tag_constants.SERVING], \
          signature_def_map = {'predictByRaw': pred_signature, \
                               'predictById' : pred_signature_id, \
                               'serving_default':pred_signature_id}, \
          strip_default_attrs = True, \
          clear_devices = True, \
          main_op= tf.tables_initializer())  ## here notice ##
  builder.save()
  #builder.save(as_text=True) ## here save the pbtxt, will be suit to read by Hand #
  print getNow(), 'export model done'
