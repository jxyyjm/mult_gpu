#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
	非feed流下，训练与预测；
	多GPU-Test.
	加入 learning-decay, tf.train.exponential_decay)
	tf.layers.batch_normalization
	tf.clip_by_global_norm
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

regularizer = tf.contrib.layers.l2_regularizer(0.01)
model_dir   = './model_v5'
predict_flag = sys.argv[1]
if predict_flag == 'False':
	dataset_flow = get_batch_data(train_data_file, buffer_size=1000, batch_size=64, epoch=1000, shuffle=True)
	x, y         = dataset_flow
	training=True
	with tf.variable_scope('layer-1', reuse = tf.AUTO_REUSE):
		y1 = tf.layers.dense(inputs = x, units = 128, use_bias=False, \
		                activation = tf.nn.relu, \
			            kernel_regularizer = regularizer, \
       	       	        bias_regularizer = None)
		y1 = tf.layers.batch_normalization(y1, training= training)
	with tf.variable_scope('layer-2', reuse = tf.AUTO_REUSE):
		y2 = tf.layers.dense(inputs = y1, units = 64, use_bias=False, \
						activation = tf.nn.relu, \
						kernel_regularizer = regularizer, \
						bias_regularizer = None)
		y2 = tf.layers.batch_normalization(y2, training= training)
	with tf.variable_scope('layer-3', reuse = tf.AUTO_REUSE):
		y3 = tf.layers.dense(inputs = y2, units = 2, use_bias=True, \
						activation = None, \
						kernel_regularizer = regularizer, \
						bias_regularizer = regularizer)

	logits     = y3
	prob_all   = tf.nn.softmax(logits, 1)
	pred_class = tf.argmax(prob_all, 1)
	prob_class = tf.reduce_max(prob_all, 1)
	prob_class_ = tf.cast(tf.expand_dims(prob_class, 1), dtype=tf.float32)
	pred_class_ = tf.cast(tf.expand_dims(pred_class, 1), dtype=tf.float32)
	real_label_ = tf.cast(tf.expand_dims(tf.argmax(y, 1), 1), dtype=tf.float32)
	res_merge_  = tf.concat([prob_class_, pred_class_, real_label_], 1)

	accuracy    = tf.contrib.metrics.accuracy(labels = tf.argmax(y,1), predictions = pred_class)
	confusion   = tf.contrib.metrics.confusion_matrix(labels = tf.argmax(y,1),predictions = pred_class)

	with tf.variable_scope(tf.get_variable_scope()):
		global_step = tf.train.get_or_create_global_step()
		#lr_decay   = tf.train.exponential_decay(learning_rate=0.004, global_step=global_step, decay_steps=5000, decay_rate=0.99, staircase = True)
		lr_decay   = 0.004
		optimizer  = tf.train.AdamOptimizer(learning_rate = lr_decay)
		tower_grads = []
		tower_logits= []
		for i in xrange(4):
			with tf.device('/gpu:' +str(i)):
				with tf.name_scope('name_scope-'+str(i)) as scope:
					print 'gpu:', i, 'tf.global_variables', tf.global_variables()
					tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
					#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope) ## 这里是获取不到update_ops的，因为将网络定义到了CPU上，不再这个scope下 ##
					update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
					print 'gpu:', i, 'update_ops :', update_ops
					with tf.control_dependencies(update_ops): 
						losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
						print 'gpu:', i, 'losses is :', losses 
						total_loss = tf.add_n(losses, name='total_loss')
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
			while True:
				try:
					_, loss_ = sess.run([train_op, total_loss])
				except tf.errors.OutOfRangeError:
					print 'train end'
					break
				cur_step = tf.train.global_step(sess, global_step)
				if cur_step % 100== 0: start_time = time.time()
				if cur_step % 100== 1:
					accu         = sess.run(accuracy)
					summary_res  = sess.run(merged_summary)
					writer.add_summary(summary_res, cur_step)
					duration     = time.time() - start_time
					print 'iter:\t', cur_step, '\tloss:\t', loss_, '\taccuracy:\t', accu, '\ttime cost(sec):\t', duration
				if cur_step % 200==1:
					print 'iter:', cur_step, 'in Graph     , layer-1/bn/moving_mean', sess.run('layer-1/batch_normalization/moving_mean:0')[0]
					print 'iter:', cur_step, 'in Graph     , layer-1/bn/moving_variance', sess.run('layer-1/batch_normalization/moving_variance:0')[0]
					print 'iter:', cur_step, 'in Graph     , layer-1/bn/beta', sess.run('layer-1/batch_normalization/beta:0')[0]
					print 'iter:', cur_step, 'in Graph     , layer-1/bn/gamma', sess.run('layer-1/batch_normalization/gamma:0')[0]
				if cur_step % 10000==1:
					print 'save model into path:', model_dir, cur_step
					saver.save(sess, model_dir+'/ckpt', global_step=cur_step)
else:
	in_x, in_y = get_batch_data(test_data_file, shuffle=False, batch_size=1000, epoch=1, drop_last=False)
	training  = False
	with tf.variable_scope('layer-1', reuse = tf.AUTO_REUSE):
		y1 = tf.layers.dense(inputs = in_x, units = 128, use_bias=False, \
			                activation = tf.nn.relu, \
				            kernel_regularizer = regularizer, \
           	       	        bias_regularizer = None)
		y1 = tf.layers.batch_normalization(y1, training= training)
	with tf.variable_scope('layer-2', reuse = tf.AUTO_REUSE):
		y2 = tf.layers.dense(inputs = y1, units = 64, use_bias=False, \
							activation = tf.nn.relu, \
							kernel_regularizer = regularizer, \
							bias_regularizer = None)
		y2 = tf.layers.batch_normalization(y2, training= training)
	with tf.variable_scope('layer-3', reuse = tf.AUTO_REUSE):
		y3 = tf.layers.dense(inputs = y2, units = 2, use_bias=True, \
							activation = None, \
							kernel_regularizer = regularizer, \
							bias_regularizer = regularizer)
	logits = y3		
	prob_all   = tf.nn.softmax(logits, 1)
	pred_class = tf.argmax(prob_all, 1)
	prob_class = tf.reduce_max(prob_all, 1)
	
	prob_class = tf.cast(tf.expand_dims(prob_class, 1), dtype=tf.float32)
	pred_class = tf.cast(tf.expand_dims(pred_class, 1), dtype=tf.float32)	
	real_label = tf.cast(tf.expand_dims(tf.argmax(in_y, 1), 1), dtype=tf.float32)
	pred_res   = tf.concat([prob_class, pred_class, real_label], 1)
	saver = tf.train.Saver()
	ckpt_file = tf.train.latest_checkpoint(model_dir)
	print 'here ckpt_file:', ckpt_file
	if ckpt_file:
		config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True), device_count = {'GPU':0}, allow_soft_placement = True)
		with tf.Session(config=config) as sess:
			saver.restore(sess, ckpt_file)
			sess.run(tf.tables_initializer())
			global_variables = tf.global_variables()
			print 'tf.global_variables', global_variables
			for i in global_variables: print i.name
			from tensorflow.python import pywrap_tensorflow
			reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
#			## 这里验证了模型里的参数被加载正确 ##
			print 'in model_save, layer-1/bn/moving_mean', reader.get_tensor('layer-1/batch_normalization/moving_mean')
			print 'in Graph     , layer-1/bn/moving_mean', sess.run('layer-1/batch_normalization/moving_mean:0')
#			print 'in model_save, layer-1/bn/moving_variance', reader.get_tensor('layer-1/batch_normalization/moving_variance')
#			print 'in Graph     , layer-1/bn/moving_variance', sess.run('layer-1/batch_normalization/moving_variance:0')
#			print 'in model_save, layer-1/bn/beta', reader.get_tensor('layer-1/batch_normalization/beta')
#			print 'in Graph     , layer-1/bn/beta', sess.run('layer-1/batch_normalization/beta:0')
#			print 'in model_save, layer-1/bn/gamma', reader.get_tensor('layer-1/batch_normalization/gamma')
#			print 'in Graph     , layer-1/bn/gamma', sess.run('layer-1/batch_normalization/gamma:0')
#			print 'in model_save, layer-2/bn/moving_mean', reader.get_tensor('layer-2/batch_normalization/moving_mean')
#			print 'in Graph     , layer-2/bn/moving_mean', sess.run('layer-2/batch_normalization/moving_mean:0')
#			print 'in model_save, layer-2/bn/moving_variance', reader.get_tensor('layer-2/batch_normalization/moving_variance')
#			print 'in Graph       layer-2/bn/moving_variance', sess.run('layer-2/batch_normalization/moving_variance:0')
			right   = 0
			all_num = 0
			iter_num= 0
			while True:
				try:
					res = sess.run(pred_res)
					right += sess.run(tf.reduce_sum(tf.cast(tf.equal(res[:,1], res[:,2]), tf.int32)))
					all_num+= res.shape[0]
					#print res
				except tf.errors.OutOfRangeError:
					print 'pred end'
					break
				if iter_num % 10 == 0: ## 这里查看图中的变量，是一致未有变化的 ##验证 training=False下，BN的学习到的系数都是不动的 ##
#					print 'iter:', iter_num, 'in Graph     , layer-1/bn/moving_mean', sess.run('layer-1/batch_normalization/moving_mean:0')
#					print 'iter:', iter_num, 'in Graph     , layer-1/bn/moving_variance', sess.run('layer-1/batch_normalization/moving_variance:0')
#					print 'iter:', iter_num, 'in Graph     , layer-2/bn/moving_mean', sess.run('layer-2/batch_normalization/moving_mean:0')
#					print 'iter:', iter_num, 'in Graph       layer-2/bn/moving_variance', sess.run('layer-2/batch_normalization/moving_variance:0')
					
					print 'iter:', iter_num, 'in Graph     , layer-1/bn/beta', sess.run('layer-1/batch_normalization/beta:0')
					print 'iter:', iter_num, 'in Graph     , layer-1/bn/gamma', sess.run('layer-1/batch_normalization/gamma:0')
					iter_num += 1	
			print 'right-num:', right, 'all_num:', all_num, 'accuracy:', float(right)/float(all_num)
	else:
		print 'no check point in path:', model_dir
	
		
