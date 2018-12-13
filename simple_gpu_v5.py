#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
	非feed流下，训练与预测；
	多GPU-Test.
	加入 learning-decay, tf.train.exponential_decay)
	tf.layers.batch_normalization
	tf.clip_by_global_norm
	DCN as model
	解决掉了随着迭代次数增加，batch_cost time 也增加的问题 #3
	将图训练和预测彻底分拆，需要思考，如何构造图及其预测 加载 ## 
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import time
import numpy as np
import tensorflow as tf
from base import average_gradients

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket']
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

train_data_file = './data/adult.data'
test_data_file  = './data/adult.test'

def get_batch_data(file_name, batch_size=10, buffer_size=100, epoch=10, shuffle=True, drop_last=True, embedding_map=None, reader=None):
	columns = build_model_columns(embedding_map, reader)
	def decode_line(line):
		columns = tf.decode_csv(line, \
				record_defaults=_CSV_COLUMN_DEFAULTS)
		return dict(zip(_CSV_COLUMNS, columns))
		#return columns[:-1], columns[-1:]
		## 注意record_defaults会作为默认数据类型去检查field的内容 ##
	def tensor_from_input_layer(input, columns):
		return tf.feature_column.input_layer( \
					features = input, \
					feature_columns = columns, \
					trainable = True)
		## 注意，这个函数在dataset流里处理是快的，不要放到session里面执行，会超级慢 ##
	dataset = tf.contrib.data.TextLineDataset(file_name)
	dataset = dataset.map(decode_line, num_threads = 5)
	if shuffle: dataset = dataset.shuffle(buffer_size = buffer_size)
	dataset = dataset.repeat(count = epoch)
	if drop_last == True:
		dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
	else:
		dataset = dataset.batch(batch_size = batch_size)
	dataset = dataset.prefetch(batch_size*2)
	dataset = dataset.make_one_shot_iterator()
	feature = dataset.get_next()
	labels  = tf.equal(feature.pop('income_bracket'), '>50K')
	labels  = tf.reshape(labels, [-1])
	transFea= tensor_from_input_layer(feature, columns)
	transLab= tf.one_hot(tf.cast(labels, tf.int32), 2)
	print 'feature:',  feature
	print 'transFea:', transFea
	print 'labels:',   labels
	print 'transLab:', transLab
	return transFea, transLab

def build_model_columns(embedding_map=None, reader=None):
  # Continuous columns
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')
  education = tf.feature_column.categorical_column_with_vocabulary_list(
          'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
          'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
          'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])
  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
          'workclass', ['?', 'Federal-gov', 'Local-gov', 'Never-worked',
          'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'human'])
  occupation = tf.feature_column.categorical_column_with_vocabulary_list(
          'occupation',['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
          'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
          'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'])
  age_buckets = tf.feature_column.bucketized_column(
       age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  em_workclass= tf.feature_column.embedding_column(workclass,  dimension=5, combiner='mean', initializer=None)
  em_education= tf.feature_column.embedding_column(education,  dimension=5, combiner='mean', initializer=None)
  em_occupation=tf.feature_column.embedding_column(occupation, dimension=5, combiner='mean', initializer=None)
  
  columns = [
      age,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      em_workclass,
      em_education,
      em_occupation]
  return columns

training     = True
regularizer  = tf.contrib.layers.l2_regularizer(0.01)
model_dir    = './model_v6'
dataset_flow = get_batch_data(train_data_file, buffer_size=1000, batch_size=64, epoch=1000, shuffle=True)
x, y         = dataset_flow

def cross_op(x0, x, w, b):
	x0 = tf.expand_dims(x0, axis=2)
	x  = tf.expand_dims(x,  axis=2)
	dot= tf.matmul(x0, tf.transpose(x, [0, 2, 1]))
	mid_res = tf.multiply(dot, tf.transpose(w))
	res     = tf.reduce_sum(mid_res, axis=2) + tf.transpose(b)
	return res

with tf.variable_scope('dnn-layer-1', reuse = tf.AUTO_REUSE):
	y1 = tf.layers.dense(inputs = x, units = 128, use_bias=False, \
						activation = tf.nn.relu, \
						kernel_regularizer = None, \
						bias_regularizer = None)
	y1 = tf.layers.batch_normalization(y1, training= training)
with tf.variable_scope('dnn-layer-2', reuse = tf.AUTO_REUSE):
	y2 = tf.layers.dense(inputs = y1, units = 64, use_bias=False, \
						activation = tf.nn.relu, \
						kernel_regularizer = None, \
						bias_regularizer = None)
	y2 = tf.layers.batch_normalization(y2, training= training)
with tf.variable_scope('dnn-layer-3', reuse = tf.AUTO_REUSE):
	y3 = tf.layers.dense(inputs = y2, units = 32, use_bias=True, \
						activation = None, \
						kernel_regularizer = None, \
						bias_regularizer = None)
	dnn_output = y3
#		column_num = x.get_shape().as_list()[1]
#		with tf.variable_scope('cross-layer-1', reuse= tf.AUTO_REUSE):
#			c_w_1  = tf.get_variable(name='w1', initializer=tf.random_normal((column_num, 1), mean=0.0, stddev=0.02), dtype=tf.float32, regularizer= regularizer)
#			c_b_1  = tf.get_variable(name='b1', initializer=tf.zeros((column_num, 1), dtype=tf.float32))
#			c_y_1  = cross_op(x, x, c_w_1, c_b_1)
#		with tf.variable_scope('cross-layer-2', reuse= tf.AUTO_REUSE):
#			c_w_2  = tf.get_variable(name='w2', initializer=tf.random_normal((column_num, 1), mean=0.0, stddev=0.02), dtype=tf.float32, regularizer= regularizer)
#			c_b_2  = tf.get_variable(name='b2', initializer=tf.zeros((column_num, 1), dtype=tf.float32))
#			c_y_2  = cross_op(x, c_y_1, c_w_2, c_b_2)
#			cro_output = c_y_2
with tf.variable_scope('merge-layer', reuse= tf.AUTO_REUSE):	
#			merge = tf.concat([cro_output, dnn_output], 1)
	merge = dnn_output
	output= tf.layers.dense(inputs = merge, units = 2, use_bias = True, \
							activation = None, \
							kernel_regularizer= regularizer, \
							bias_regularizer=None, name = 'output')
logits     = output
prob_all   = tf.nn.softmax(logits, 1)
pred_class = tf.argmax(prob_all, 1)
prob_class = tf.reduce_max(prob_all, 1)
prob_class_ = tf.cast(tf.expand_dims(prob_class, 1), dtype=tf.float32)
pred_class_ = tf.cast(tf.expand_dims(pred_class, 1), dtype=tf.float32)
real_label_ = tf.cast(tf.expand_dims(tf.argmax(y, 1), 1), dtype=tf.float32)
res_merge_  = tf.concat([prob_class_, pred_class_, real_label_], 1)

accuracy    = tf.contrib.metrics.accuracy(labels = tf.argmax(y,1), predictions = pred_class)
confusion   = tf.contrib.metrics.confusion_matrix(labels = tf.argmax(y,1),predictions = pred_class)

#with tf.variable_scope(tf.get_variable_scope()):
global_step = tf.train.get_or_create_global_step()
lr_decay    = tf.train.exponential_decay(learning_rate=0.008, global_step=global_step, decay_steps=1000, decay_rate=0.99, staircase = True)
optimizer   = tf.train.AdamOptimizer(learning_rate = lr_decay)
tower_grads = []
tower_logits= []
for i in xrange(4):
	with tf.device('/gpu:' +str(i)):
		with tf.name_scope('name_scope-'+str(i)) as scope:
			print 'gpu:', i, 'tf.global_variables', tf.global_variables()
			tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
			with tf.control_dependencies(update_ops): 
				losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
				print 'gpu:======', i, 'losses is :', losses 
				regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) ## 疑问？这里没法在scope获取到权重的正则化值 ##是因为大家都是共享的权重，其当前的正则化应该也是一致的才对 #
				print 'gpu:======', i, 'regular loss is:', regular_losses
				total_loss = tf.add_n(losses + regular_losses, name='total_loss')
				grads  = optimizer.compute_gradients(total_loss)
				#gradients, variables = zip(*optimizer.compute_gradients(total_loss))
				#gradients, _ = tf.clip_by_global_norm(t_list=gradients, clip_norm=100.0) ## clip ##
				#grads   = zip(gradients, variables)
				tf.summary.scalar('loss', total_loss) ## 这里对每个name_scope下的loss都做了记录
			tower_grads.append(grads)
			tower_logits.append(logits)
grads = average_gradients(tower_grads)
train_op = optimizer.apply_gradients(grads, global_step=global_step) ## 每次执行到这里，会对变量global_step自增1 #

merged_summary = tf.summary.merge_all()
saver  = tf.train.Saver()
config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True), device_count = {'GPU':4}, allow_soft_placement = True)
with tf.Session(config=config) as sess:
	writer = tf.summary.FileWriter(model_dir, sess.graph)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.tables_initializer())
	start_time = time.time()
	sess.graph.finalize()
	while True:
		try:
			_, loss_ = sess.run([train_op, total_loss])
		except tf.errors.OutOfRangeError:
			print 'train end'
			break
		cur_step = tf.train.global_step(sess, global_step)
		if cur_step % 100== 0: start_time = time.time()
		if cur_step % 100== 1:
			accu, conf_  = sess.run([accuracy, confusion]) ## self.eval() 会添加新的op到图中 ## 因为最开始的时候没有做过eval的op ##
			summary_res  = sess.run(merged_summary)
			writer.add_summary(summary_res, cur_step)
			duration     = time.time() - start_time
			print 'iter:\t', cur_step, '\tloss:\t', loss_, '\taccuracy:\t', accu, '\ttime cost(sec):\t', duration
		if cur_step % 10000==1:
			print 'save model into path:', model_dir, cur_step
			saver.save(sess, model_dir+'/ckpt', global_step=cur_step)
