#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
	探讨非feed流的形式下，如何训练与预测
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf

iris_data_file = './iris.data'
def get_batch_data(file_name, batch_size=10, buffer_size=100, epoch=10, shuffle=True, drop_last=True):
	# return dataset.get_next() #
	columns = build_column_1()
	def decode_line(line):
		columns = tf.decode_csv(line, \
				record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])
		return dict(zip(['feat1', 'feat2', 'feat3', 'feat4'], columns[:-1])), columns[-1:]
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
	#dataset = dataset.make_initializable_iterator()
	x, y = dataset.get_next()
	y = tf.one_hot(y, 3)
	#print 'dataset y.get_shape', y.get_shape()
	y = tf.squeeze(y, 1)
	return tensor_from_input_layer(x, columns), y

def build_column_1():
	feat1_column = tf.feature_column.numeric_column('feat1')
	feat2_column = tf.feature_column.numeric_column('feat2')
	feat3_column = tf.feature_column.numeric_column('feat3')
	feat4_column = tf.feature_column.numeric_column('feat4')
	feature_columns = [feat1_column, feat2_column, feat3_column, feat4_column]
	return feature_columns
class net_model:
	def __init__(self):
		self.regularizer = tf.contrib.layers.l2_regularizer(0.01)
		with tf.variable_scope('layer-1', reuse=tf.AUTO_REUSE):
			self.w1 = tf.get_variable('w1', initializer = tf.random_normal(shape = [4,8], mean  = 0.0, stddev= 1.0), regularizer=self.regularizer, dtype=tf.float32)
			self.b1 = tf.get_variable('b1', initializer = tf.zeros([1,8]), dtype = tf.float32)
		with tf.variable_scope('layer-2', reuse=tf.AUTO_REUSE):
			self.w2 = tf.get_variable('w2', initializer = tf.random_normal(shape = [8,3], mean  = 0.0, stddev= 1.0), regularizer=self.regularizer, dtype=tf.float32)
			self.b2 = tf.get_variable('b2', initializer = tf.zeros([1,3]), dtype = tf.float32)
	def logits(self, x):
		y1= tf.matmul(x, self.w1) + self.b1
		y2= tf.matmul(y1,self.w2) + self.b2
		return y2
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
	def train(self):
		save = tf.train.Saver()
		with tf.Session() as sess:
			in_x, in_y = get_batch_data(iris_data_file, buffer_size=100, batch_size=30, epoch=1000, shuffle=True)
			optimizer  = tf.train.AdamOptimizer(learning_rate = 0.001)
			logits     = self.logits(in_x)
			loss       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=in_y, logits=logits))
			train_op   = optimizer.minimize(loss)
			sess.run(tf.global_variables_initializer())
			sess.run(tf.tables_initializer())
			iter_num = 0
			while True:
				try:
					_, loss_ = sess.run([train_op, loss])
					if iter_num % 100==1:
						accu, _ = sess.run(self.eval(in_x, in_y))
						print 'iter:', iter_num, loss_, accu
					iter_num += 1
					if iter_num % 1000==1:
						print 'save model into path:', model_dir, iter_num
						saver.save(sess, model_dir+'/ckpt', global_step=iter_num)
				except tf.errors.OutOfRangeError:
					print 'train end'
					break
	def predict(self, file_or_list, sess):
		data_flow = get_batch_data(file_or_list, shuffle=False, batch_size=12, epoch=1, drop_last=False)
		in_x, in_y= data_flow
		while True:
			try:
				res = sess.run(self.infernce(in_x, in_y))
				print res
			except tf.errors.OutOfRangeError:
				print 'pred end'
				break
def main():	
	predict_flag = True
	model_dir    = './model'
	net_used     = net_model()
	if predict_flag == False:
		net_used.train()
	else:
		saver = tf.train.Saver()
		ckpt_file = tf.train.latest_checkpoint(model_dir)
		if ckpt_file:
			with tf.Session() as sess:
				saver.restore(sess, ckpt_file)
				print 'net_used.para is the saved para-values'
				print 'net_used.w1', sess.run(net_used.w1)
				print 'net_used.b1', sess.run(net_used.b1)
				print 'net_used.w2', sess.run(net_used.w2)
				print 'net_used.b2', sess.run(net_used.b2)
				net_used.predict(iris_data_file, sess)
		else:
			print 'no check point in path:', model_dir

if __name__=='__main__':
	main()
