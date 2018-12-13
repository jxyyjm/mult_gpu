#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf


with tf.variable_scope('foo'):
  w = tf.Variable(tf.ones((2,3), dtype=tf.float32), name='w')
  a = tf.get_variable('a', [1])
  with tf.variable_scope('hk', reuse=tf.AUTO_REUSE):
    b = tf.get_variable('b', [1])
    c = tf.get_variable('b', [1])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print 'w', w.name
print 'a', a.name
print 'b', b.name
print 'c', c.name
print '-'*10, 'w', '-'*10
print sess.run(w)
print '-'*10, 'a', '-'*10
print sess.run(a)
print '-'*10, 'b', '-'*10
print sess.run(b)
print '-'*10, 'c', '-'*10
print sess.run(c)
