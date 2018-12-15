#!/usr/bin/python
# -*- coding:utf-8 -*-
# tf 1.4 #

import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
'''
	## notice: Important ##
	## 关于变量的复用，建议使用tf.variable_scope(name, reuse=tf.AUTO_REUSE):
	## 可以自动处理，存在变量时则复用，不存在变量时则创建 ##
	## 手动设置，tf.variable_scope.reuse容易出现变量不存在的错误 ##
'''

'''
print '-'*10, 'this is is test scope.reuse'
with tf.variable_scope('scope1') as scope1:
	var1 = tf.get_variable('var1', [1])
	scope1.reuse_variables() ## set the scope1.reuse=True ##
	var2 = tf.get_variable('var1', [1])
	print 'current scope trainable_variables:', scope1.trainable_variables() ## here 发现只有一个可训练变量
	print 'current scope :', scope1
	print 'current scope :', tf.get_variable_scope(), scope1
print var1.name
print var2.name

print '-'*10, 'this is test scope.reuse'
with tf.variable_scope('scope3') as scope3:
	var3 = tf.get_variable('var3', [1])
	print 'var3.name', var3.name
	print 'first layer, current scope:', scope3
	print 'first layer, current scope:', tf.get_variable_scope()
	print 'first layer, scope  trainable_variables:', tf.get_variable_scope().trainable_variables()
	print 'first layer, scope  trainable_variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scope3') ## 只能对trainable_variables的name的第一个前缀做过滤 ##
	print 'first layer, global trainable_variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	with tf.variable_scope('scope4') as scope4:
		var4 = tf.get_variable('var4', [1])
		print 'var4.name', var4.name ## scope3/scope4/var4:0 ##
		print 'secon layer, current scope:', scope4
		print 'secon layer, current scope:', tf.get_variable_scope()
		print 'secon layer, vari_scope  trainable_variables:', tf.get_variable_scope().trainable_variables()
		print 'secon layer, name_scope  trainable_variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scope3')
		print 'secon layer, global      trainable_variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		scope4.reuse_variables()
		var5 = tf.get_variable('var4', [1]) ## var5.name == scope/scope4/var4:0 ##
		var6 = tf.get_variable('var4', [1]) 
		## 如果是var6=tf.get_variable('var5', [1]) 会报错，
		## 因为什么呢？scop4.reuse_variables() 已经设定了 
		## 那么tf.get_variabel只会寻找同名的,没有同名地只好报错啦 ##
		## var7 = tf.get_variable('var3', [1]) ## 这里也会报错，
		## 是因为 scope4做了 reuse=True, 但是这个variable_scope并没有变量名称为var4的变量。
		with tf.variable_scope(scope3, reuse=tf.AUTO_REUSE):
			## 这种方式比较靠谱，会自动处理reuse 和 变量存在的问题 
			## 检查域内，有同名变量则复用，无同名变量则创建
			var7 = tf.get_variable('var3', [1])
			var8 = tf.get_variable('var8', [1])
			## var7.name == scope3/var3:0 ##
			## var8.name == scope3/var8:0 ## 
		print 'var5.name', var5.name
		print 'var6.name', var6.name
		print 'var7.name', var7.name
		print 'var8.name', var8.name
		print 'third layer, current scope:', scope4
		print 'third layer, current scope:', tf.get_variable_scope()
		print 'third layer, vari_scope  trainable_variables:', tf.get_variable_scope().trainable_variables()
		print 'third layer, name_scope  trainable_variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scope3')
		print 'third layer, global      trainable_variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
'''

'''
print '-'*50, 'next test'
def my_layer(x):
	print 'in my_layer, tf.get_variable_scope', tf.get_variable_scope()
	with tf.variable_scope('layer_1', reuse=tf.AUTO_REUSE):
		w1 = tf.get_variable('w1', [3, 4])
		print '    w1.name', w1.name
	with tf.variable_scope('layer_2', reuse=tf.AUTO_REUSE):
		w2 = tf.get_variable('w2', [4,1])
		print '    w2.name', w2.name
	y = tf.matmul(tf.matmul(x, w1), w2)
	return y

with tf.variable_scope(tf.get_variable_scope()) as scope_used:
	print 'tf.get_variable_scope', tf.get_variable_scope()
	with tf.name_scope('class') as scope_here:
		x = tf.random_normal([2,3])
		y1= my_layer(x)
		print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'layer')
		y2= my_layer(x)
		print 'y1.name', y1.name
		print 'y2.name', y2.name
		print 'layer 下的trainable variables :', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'layer') 
		## 这里虽然有name_scope，调用了两次my_layer, 但是仍然复用layer-1/w1 和 layer-2/w2
	with tf.variable_scope('another'):
		y3 = my_layer(x)
		y4 = my_layer(x)
		print 'y3.name', y3.name
		print 'y4.name', y4.name
		print 'another 下的trainable variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'another')
		## 这里是另一层variable_scope，在里面调用了my_layer函数，那么my_layer函数里面的variable_scope就是下一级variable_scope ## var.name is another/layer-?/w? ##
	## 这里查看整体的可训练变量，发现是两组，在不同的命名空间内another/layer-1/w1; another/layer-2/w2; 和 layer-1/w1; layer-2/w2
	## 每组参数，都被复用了 ##
	## 变量域，设置了reuse=tf.AUTO_REUSE，则每次使用到这个变量域时，会自动复用 
	## 			有同名变量则复用，无同名变量则创建 ## 名：指 用variable_scope1/variable_scope2/weight 这样的来构成的 ##
	##			同名检查，是包括这些变量域前缀的检查的 ## 必须是同域且同前缀的，才认为是同名变量 ##
	## notice: 变量是以variable_scope来作name前缀的 ##
	## notice: operation是以name_scope来作name前缀的 ##
	print '-'*10, 'all trainable variables:', '-'*10, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
'''
'''
print '-'*50, 'next action'

def my_layer(x):
	print 'in my_layer, tf.get_variable_scope', tf.get_variable_scope()
	with tf.variable_scope('layer_1', reuse=tf.AUTO_REUSE):
		w1 = tf.get_variable('w1', [3, 4])
		print '    w1.name', w1.name
	with tf.variable_scope('layer_2', reuse=tf.AUTO_REUSE):
		w2 = tf.get_variable('w2', [4,1])
		print '    w2.name', w2.name
	y = tf.matmul(tf.matmul(x, w1), w2)
	return y

with tf.variable_scope(tf.get_variable_scope()):
	for i in xrange(2):
		with tf.device('/gpu:'+str(i)): ## 在device上，做下面的这些操作 ##
			print 'gpu:', i, '='*10
			with tf.name_scope('name_scope_'+str(i)) as scope:
				x  = tf.random_normal([2,3]) 
				y1 = my_layer(x)
				y2 = my_layer(x)
				print 'y1.name', y1.name
				print 'y2.name', y2.name
			print 'when gpu:', i, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			## 这里发现，所有的重复调用该函数，以及多个GPU重复调用该函数，
			## 都没有再额外创建变量，仅有的变量是layer-1/w1; layer-2/w2 ## 不管在哪个GPU上时，都是共享这两个变量 ##
			## 该函数my_layer，已经设置过 reuse=tf.AUTO_REUSE)了，自动处理存在的变量作共享变量 ##
			## 并且，在调用函数时，没有再额外包variable_scope，都是同层关系，所以同层variable-scope内共享了##
print 'last:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
'''


print '-'*50, 'check name_scope variable_scope tf.get_collection( , scope) relation' 


def my_layer(x):
	print 'in my_layer, tf.get_variable_scope', tf.get_variable_scope()
	with tf.variable_scope('layer_1', reuse=tf.AUTO_REUSE):
		w1 = tf.get_variable('w1', [3, 4])
		print '    w1.name', w1.name
	with tf.variable_scope('layer_2', reuse=tf.AUTO_REUSE):
		w2 = tf.get_variable('w2', [4,1])
		print '    w2.name', w2.name
	y = tf.matmul(tf.matmul(x, w1), w2)
	return y

with tf.name_scope('name_scope') as name_scope:
	x1 = tf.random_normal([2,3])
	y1 = my_layer(x1)
	x2 = tf.random_normal([2,3])
	y2 = my_layer(x2)
	y  = y2 - y1
	print 'x1.name', x1.name
	print 'y1.name', y1.name
	print 'x2.name', x2.name
	print 'y2.name', y2.name
	print 'y.name ',  y.name
	print 'update_ops:', tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	print 'global_variables:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	print 'model_variables:', tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
	print 'summaries:', tf.get_collection(tf.GraphKeys.SUMMARIES)
	print ''
	print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

	
    
	
