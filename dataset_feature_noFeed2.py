#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
	非feed流下，
	测试 tf.feature_column里面的参数是否也会被恢复出来 ##
	ok, 可以完美地恢复出来 ## 
	预先将所有变量定义，然后再restore，避免只预定义了部分(比如dataset流里的，也会有变量在tf.GraphKeys.TRABLE_VARIABLES)
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
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

class net_model:
	def __init__(self, model_dir):
		self.regularizer = tf.contrib.layers.l2_regularizer(0.01)
		with tf.variable_scope('layer-1', reuse=tf.AUTO_REUSE):
			self.w1 = tf.get_variable('w1', initializer = tf.random_normal(shape = [32,128], mean  = 0.0, stddev= 1.0), regularizer=self.regularizer, dtype=tf.float32)
			self.b1 = tf.get_variable('b1', initializer = tf.zeros([1,128]), dtype = tf.float32)
		with tf.variable_scope('layer-2', reuse=tf.AUTO_REUSE):
			self.w2 = tf.get_variable('w2', initializer = tf.random_normal(shape = [128,64], mean  = 0.0, stddev= 1.0), regularizer=self.regularizer, dtype=tf.float32)
			self.b2 = tf.get_variable('b2', initializer = tf.zeros([1,64]), dtype = tf.float32)
		with tf.variable_scope('layer-3', reuse=tf.AUTO_REUSE):
			self.w3 = tf.get_variable('w3', initializer = tf.random_normal(shape = [64,2], mean  = 0.0, stddev= 1.0), regularizer=self.regularizer, dtype=tf.float32)
			self.b3 = tf.get_variable('b3', initializer = tf.zeros([1,2]), dtype = tf.float32)	
		self.model_dir = model_dir

	def logits(self, x):
		y1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
		y2 = tf.nn.relu(tf.matmul(y1,self.w2) + self.b2)
		y3 = tf.matmul(y2, self.w3) + self.b3
		return y3
	def infernce(self, x, y=None):
		logits     = self.logits(x)
		prob_all   = tf.nn.softmax(logits, 1)
		pred_class = tf.argmax(prob_all, 1)
		prob_class = tf.reduce_max(prob_all, 1)
		if y is not None:
			prob_class = tf.cast(tf.expand_dims(prob_class, 1), dtype=tf.float32)
			pred_class = tf.cast(tf.expand_dims(pred_class, 1), dtype=tf.float32)
			real_label = tf.cast(tf.expand_dims(tf.argmax(y, 1), 1), dtype=tf.float32)
			return tf.concat([prob_class, pred_class, real_label], 1)
		else:
			prob_class = tf.cast(tf.expand_dims(prob_class, 1), dtype=tf.float32)
			pred_class = tf.cast(tf.expand_dims(pred_class, 1), dtype=tf.float32)	
			return tf.concat([prob_class, pred_lcass], 1)
	def eval(self, x, y):
		labels      = tf.argmax(y, 1)
		logits      = self.logits(x)
		pred_class  = tf.argmax(tf.nn.softmax(logits, 1), 1)
		accuracy    = tf.contrib.metrics.accuracy(labels = labels, predictions = pred_class)
		confusion   = tf.contrib.metrics.confusion_matrix(labels = labels, predictions = pred_class)
		return (accuracy, confusion)
	def train(self, dataset_flow):
		save = tf.train.Saver()
		with tf.Session() as sess:
			in_x, in_y = dataset_flow
			optimizer  = tf.train.AdamOptimizer(learning_rate = 0.008)
			logits     = self.logits(in_x)
			loss       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=in_y, logits=logits))
			train_op   = optimizer.minimize(loss)
			sess.run(tf.global_variables_initializer())
			sess.run(tf.tables_initializer())
			iter_num = 0
			saver = tf.train.Saver()
			while True:
				try:
					_, loss_ = sess.run([train_op, loss])
					if iter_num % 100==1:
						accu, _ = sess.run(self.eval(in_x, in_y))
						print 'iter:', iter_num, loss_, accu
					iter_num += 1
					if iter_num % 2000==1:
						print 'save model into path:', self.model_dir, iter_num
						saver.save(sess, self.model_dir+'/ckpt', global_step=iter_num)
				except tf.errors.OutOfRangeError:
					print 'train end'
					break
	def predict(self, sess, dataset_flow):
		with sess.as_default():
			in_x, in_y = dataset_flow
			right   = 0
			all_num = 0
			while True:
				try:
					res = sess.run(self.infernce(in_x, in_y))
					right += sess.run(tf.reduce_sum(tf.cast(tf.equal(res[:,1], res[:,2]), tf.int32)))
					all_num+= res.shape[0]
				except tf.errors.OutOfRangeError:
					print 'pred end'
					break
			print 'right-num:', right, 'all_num:', all_num, 'accuracy:', float(right)/float(all_num)
def main():	
	predict_flag = True
	model_dir    = './model2'
	if predict_flag == False:
		net_used = net_model(model_dir)
		dataset_flow = get_batch_data(train_data_file, buffer_size=1000, batch_size=64, epoch=1000, shuffle=True)
		net_used.train(dataset_flow)
	else:
		dataset_flow = get_batch_data(test_data_file, shuffle=False, batch_size=1000, epoch=1, drop_last=False)
		net_used = net_model(model_dir)
		saver = tf.train.Saver()
		ckpt_file = tf.train.latest_checkpoint(model_dir)
		if ckpt_file:
			with tf.Session() as sess:
				saver.restore(sess, ckpt_file)
				sess.run(tf.tables_initializer())
				net_used.predict(sess, dataset_flow)
		else:
			print 'no check point in path:', model_dir
		
		
if __name__=='__main__':
	main()
