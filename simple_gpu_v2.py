#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
  use multi-gpu in one model
  in the way of split data-array
  version-2: 以数据切分的方式，使用mult-GPU
  解决了batch_size在不定时无法切分的问题
  tensorflow >= 1.4, solve the above problem using 'batch_and_drop_remainder'
  网络结构定义模型可以单独使用自定义类
  
  奇怪的现象：
    首先：速度问题，不如直接的code model + feed流的速度, 就是没有GPU的时候也是不如其快
    其次：收敛速度很慢, 远不如直接code model + feed流脚本收敛快
    分别执行下这个脚本和 dataset_feature_feed.py
    
'''
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
from base import average_gradients

def build_column():
  feat1_column = tf.feature_column.numeric_column('feat1')
  feat2_column = tf.feature_column.numeric_column('feat2')
  feat3_column = tf.feature_column.numeric_column('feat3')
  feat4_column = tf.feature_column.numeric_column('feat4')
  feature_columns = [feat1_column, feat2_column, feat3_column, feat4_column]
  return feature_columns

iris_data_file = './iris.data'
def input_fn(file_name, epoch= 10, shuffle=False, batch_size=16):
  columns = build_column()
  def decode_line(line):
    columns = tf.decode_csv(line, \
            record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])
    #return columns[:-1], columns[-1]
    return dict(zip(['feat1', 'feat2', 'feat3', 'feat4'], columns[:-1])), columns[-1:]
  #def tensor_as_input(input, columns):
  #  return tf.feature_column.input_layer(features = input, feature_columns=columns, trainable=True)
  print 'input_fn before textlinedataset time:', time.ctime()
  #dataset = tf.contrib.data.TextLineDataset(file_name)
  dataset = tf.data.TextLineDataset(file_name)
  print 'input_fn before dataset.repeat time:', time.ctime()
  dataset = dataset.repeat(count = epoch)
  print 'input_fn before dataset.shuffle time:', time.ctime()
  if shuffle: dataset = dataset.shuffle(buffer_size = 100)
  print 'input_fn before dataset.map time:', time.ctime()
  dataset = dataset.map(decode_line, num_parallel_calls = 10)

  print 'input_fn before dataset.apply time:', time.ctime()
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  #dataset = dataset.batch(batch_size = batch_size)
  dataset = dataset.prefetch(batch_size*2)
  dataset = dataset.cache()
  print 'input_fn before shot_iterator time:', time.ctime()
  dataset = dataset.make_one_shot_iterator()
  #dataset = dataset.make_initializable_iterator() ## notice: dataset.make_one_shot_iterator是当前tf.estimator的唯一dataset模式 ##
  print 'input_fn before get_next time:', time.ctime()
  x, y = dataset.get_next()
  print 'input_fn before one_hot time:', time.ctime()
  y    = tf.one_hot(tf.squeeze(y, 1), 3)
  #print '-'*10, '#debug in dataset, as return-x:', x
  #print '-'*10, '#debug in dataset, as return-y:', y
  print 'input_fn time:', time.ctime()
  return x, y
  #return tensor_as_input(features, columns), y
class net_model:
  def __init__(self):
    print 'net_model time:', time.ctime()
    with tf.variable_scope('layer-1', reuse=tf.AUTO_REUSE): ## 命名空间内的变量是可共享的 ##
      self.w1 = tf.get_variable('w1', initializer=tf.random_normal(shape=[4,8], mean=0.0, stddev=1.0), \
                           dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.01))
      self.b1 = tf.get_variable('b1', initializer=tf.zeros(shape=[1,8], dtype=tf.float32))
      ## 可共享变量，必须用tf.get_variable()再次定义时，才会优先使用已经在前面定义过的可共享变量 ##
      ## 此时不能使用 tf.Variable() ##
    with tf.variable_scope('layer-2', reuse=tf.AUTO_REUSE):
      self.w2 = tf.get_variable('w2', initializer=tf.random_normal(shape=[8,3], mean=0.0, stddev=1.0), \
                           dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.01))
      self.b2 = tf.get_variable('b2', initializer=tf.zeros(shape=[1,3], dtype=tf.float32))
    print '#debug, in class net_model, tf.get_variable_scope', tf.get_variable_scope()
    print '#debug, in class net_model, tf.variable_scope(tf.get_variable_scope)', tf.variable_scope(tf.get_variable_scope)
    ## notice: here I set reuse=tf.AUTO_REUSE
    ## 其实在GPU上，就不用再单独设定reuse=True了 ##
  def output(self, x):
    with tf.variable_scope('compute'):
      y1 = tf.matmul(x, self.w1) + self.b1
      y2 = tf.matmul(y1,self.w2) + self.b2
    return y2
     
def mode_mine(features, labels, mode, params):
  net_used= net_model()
  print 'before tf.feature_column.input_layer time:', time.ctime()
  x       = tf.feature_column.input_layer(features=features, feature_columns=params['columns'])
  print 'after  tf.feature_column.input_layer time:', time.ctime()
  #print '-'*10, '#debug in mode_mine as input-x:', x.get_shape()
  #print '-'*10, '#debug in mode_mine as input-y:', labels.get_shape()
  ## predict mode ##
  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = net_used.output(x)
    prob        = tf.nn.softmax(logits, dim=1)
    prob_class  = tf.argmax(prob, axis=1)
    predictions_op = {'prob': prob, 'prob_class': prob_class}
    return tf.estimator.EstimatorSpec(mode, predictions=predictions_op)
  ## train and eval mode using GPU ##
  #print '#debug current dataset batch_size:', x.get_shape().as_list(), x 
  # tf.1.3无法处理最后的batch不等于batch_size的情况，这里tf.split会报错，那怎么办呢？
  # tf.1.3+可以用dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) 来做丢弃
  print 'before tf.split time:', time.ctime()
  x_list  = tf.split(x,     num_or_size_splits=params['gpu_num'], axis=0)
  y_list  = tf.split(labels,num_or_size_splits=params['gpu_num'], axis=0)
  print 'after  tf.split time:', time.ctime()
  tower_grads = []
  tower_logits= []
  optimizer   = tf.train.AdamOptimizer(learning_rate=0.002)
  print '#debug, before gpu, tf.get_variable_scope():', tf.get_variable_scope()
  print '#debug, before gpu, tf.variable_scope(tf.get_variable_scope()):', tf.variable_scope(tf.get_variable_scope())
  ## you will find, here tf.get_variable_scope 与 net_used 时，是同一个variable_scope
  with tf.variable_scope(tf.get_variable_scope()):
    ## what is happening tf.variable_scope, why the same tf.get_variable_scope as input get diff result
    ## tf.variable_scope : A context manager for defining ops that creates variables (layers) ##
    for i in xrange(params['gpu_num']):
      with tf.device('/gpu:' +str(i)):
        ## 指定在不同的GPU上，设置对应的操作 ##
        with tf.name_scope('classification-'+str(i)) as scope:
          # model and loss
          # name_scope是用来做什么的呢？
          #print '#debug, net_used.w1.name: ', net_used.w1.name
          #print '#debug, tf.get_collection(TRAINABLE_VARIABLES)', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
          #print '#debug, tf.get_variable_scope():', tf.get_variable_scope()
          ## 发现collection里面的可训练变量只有 layer-1/w1:0 这样的模型参数 ##
          #print 'gpu:', str(i), 'before net_use.output time:', time.ctime()
          logits = net_used.output(x_list[i])
          print 'gpu:', str(i), 'logits.name', logits.name
          #print '#debug, here scope:', scope, logits
          #print 'gpu:', str(i), 'before tf.losses time:', time.ctime()
          loss = tf.losses.softmax_cross_entropy(onehot_labels=y_list[i], logits=logits)
          print 'gpu:', str(i), 'loss.name', loss.name
          print 'gpu:', str(i), 'trainable_variabels', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
          ## 疑问，上面这句话为什么要单独执行 ？ 将当前name_scope的loss-opertion保存起来 ##
          ## tf.losses.softmax_cross_entrypy will do what ?
          ## 1) create a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits;
          ## 2) notice: loss_collection=tf.GraphKeys.LOSSES 这个参数
          ##    表示：collection to which the loss will be added
          ##    将这个GPU上的计算loss结果add到tf.GraphKeys.LOSSES(也即losses) 里.
          ##    其实如果有其他自定义的loss，也可以通过tf.losses.add_loss添加到collection的损失里.
          ##    背后都是用ops.add_to_collection(GraphKeys.LOSSES, loss)
          ## 3) 在来看下tf.Graph.add_to_collection(name, value)是干什么的？
          ##    store the value in the collection given by name
          ##    查看代码，最重要的一句: self._collections[name].append(value)
          ##    于是我们知道了collections是一个map，key=tf.GraphKeys, value是通过add_to_collection追加的.
          ## 难道是是为了在这个GPU下执行计算一遍loss，方便后面采集loss ##
          ## tf.losses.softmax_cross_entropy 与 tf.nn.softmax_cross_entropy_with_logits的区别 
          ## 前者处理onehot_labels 后者更适合处理2分类,且没有自动添加到tf.GraphKeys.LOSSES的动作 ## 前者处理多分类 ##
          ## 也有类似的tf.losses.sigmoid_cross_entropy 处理2分类，且将loss自动添加到tf.GraphKeys.LOSSES里 ## 
          ## 也可以使用 tf.nn.softmax_cross_entropy() 和 tf.add_to_collection(tf.GraphKeys.LOSSES, loss)来完成同样的动作 ##
          ## 疑问，为什么不直接执行 loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits) ？而要多此一举，将loss添加到tf.GraphKeys.LOSSSE里面
          ## 是因为，想要控制图的执行顺序，下面的update_ops必须先执行，再执行计算总loss的动作tf.add_n(loss_cur_gpu) ## 这个控制在BN里是非常重要的 ##
          ## 实际上，这里并不需要控制update_ops在total_loss前执行，所以可以删掉控制流逻辑的diam，total_loss直接用tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits())来计算 ##
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
          #print 'gpu:', str(i), 'update_ops.name in scop', update_ops[0].name
          #print 'gpu:', str(i), 'update_ops scope', update_ops 
          #print 'gpu:', str(i), 'update_ops all',  tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          ## 这句话，做了什么？ 从collections的map里，取出了name=Update_ops in scope下的[value]
          ## 采集 需要update的操作 ## update_ops in combination with reuse varscope
          ## explain the tf.GraphKeys.UPDATE_OPS is What.
          ## Custom functions to update some variables can be added to UPDATE_OPS
          ## and, separately run at each iteration using sess.run(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
          ## In this case, these variables are set trainable=False to avoid being updated by gradient descent.
          ## 注意这里是很重要的，会将trainable=True所涉及到的会change-variable值的操作都添加到tf.GraphKeys.UPDATE_OPS里面，
          ## 是不是，意味着tf的opt.apply_gradient是会被添加到update_ops里面？然而并不能在执行前后，发现tf.GraphKeys.UPDATE_OPS与内容，都是空的 ##？常用的是BN对参数的更新在这里更新 ##
          ## on earth, what is tf.GraphKeys.UPDATE_OPS ?
          with tf.control_dependencies(update_ops): ## 这里是做什么用的 ？## 控制流程执行顺序，update_ops先执行完毕，再计算总loss值。
            losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
            total_loss = tf.add_n(losses, name='total_loss') ## 为什么没有除以数量，求均值 ？ ##
            print '#debug# ---losses in gpu:', str(i), losses
            print '#debug# ---losses all   :', str(i), tf.get_collection(tf.GraphKeys.LOSSES) 
            print '#debug# ---trainable var:', str(i), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print '#total_loss ----- here  :', str(i), total_loss ## 这个GPU上的总loss ## 难道这里上面的loss 不是这个GPU上的总loss么，因为那个loss尺寸是[N, 1]=y.shape，并没有加和。
            #print '#debug# --- update_ops  :', str(i), update_ops
            ## tf.control_depencies表示 with段内的操作是在updates_op执行之后,再执行的 ## 控制了图的执行顺序 ##
            ## notice: 这里get_collection都使用了 scope来过滤，保证 操作和结果 都是本scope-GPU内的.
            ## 在本GPU上，根据切片输入，计算了logits，并将loss追加到collection里面；然后执行有update var的操作，再获取整个GPU上计算的loss，合起来，作为本GPU本切片的loss. 
            ## notice: 这里的total_loss是定义计算loss操作 ##
            ## 如何将对应的正则loss也提取出来 ？ ##
            ## 如果前面在变量定义时，已经用regularizer设定了，则会自动被收集到colleciton的key=regularizer_loss里面
          ## 疑惑，tf.GraphKeys.LOSSES里的值会在下次计算的时候更新么？还是继续追加呢,感觉有清理机制 ##
          ## reuse variable ##
          tf.get_variable_scope().reuse_variables() ## 将当前变量空间，设置为其中变量可以重复使用,必须配合tf.get_variable来使用 ## gpu上的操作都在同一个变量空间内，后面的gpu:1/2/3都可以复用gpu:0时使用的变量
          ## name_space 并不会影响 variable_space ##
          ## 当我的变量在定义时，都在variable_scope(reuse=tf.AUTO_REUSE)设置下，这里岂不是就可以不用了？是的 ## 当每个用到了tf.get_variable的variable_scope下，都是共享参数的 ##
          ## 这里共享的到底是谁？哪些变量？用tf.get_variable_scope来查看，发现是4个GPU下都是一样的variable_scope ##
          ## 4个GPU在同一个variable_scope下，是因为最开始的with tf.variable_scope(tf.get_variable_scope)
          ## 第一个GPU使用的trainable_variable，会由于 reuse=True的设置，允许后面GPU在使用tf.get_variable时，同名的变量是一样的/共享的.
          ## 比如，这个GPU下使用了变量layer-1/w1:0来计算loss，那么后面的GPU在使用变量layer-1/w1:0来计算loss时，是用同一个变量layer-1/w1:0
          ## 疑惑，为什么要将当前变量空间搞成变量可共享呢？都有什么变量呢？ 感觉是将w1,w2,b1,b2共享 ## yes, 就是这个样子的，如果将变量定义的地方设置为variable_scope(reuse=tf.AUTO_REUSE)，这里就省了 ##
          print '-'*10,'gpu:', i, '#debug, tf.get_variable_scope', tf.get_variable_scope()
          # grad compute
          print 'gpu:', i, 'before compute_gradients time:', time.ctime()
          grads = optimizer.compute_gradients(total_loss)
          #print '-'*10, '#debug, compute_gradients', grads
          ## this is the first part of minimize() ##
          ## optimizer.compute_gradients(loss, var_list=None, gate_gradients=GATE_OP, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)
          ## what will do this operation ? ##
          ## compute gradients of loss for the variables in var_list(default=tf.GraphKeys.TRAINABLE_VARIABLES)
          ## return a list of (gradient, variable) pair
          tower_grads.append(grads)
          tower_logits.append(logits)
  # we must calculate the mean of each gradient, notice: this is synchronization across all tower #
  print 'before average_gradient time:', time.ctime()
  grads = average_gradients(tower_grads)
  ## apply the gradients to adjust the sared variables.
  print 'before apply_gradients time:', time.ctime()
  train_op   = optimizer.apply_gradients(grads, global_step=tf.train.get_global_step())
  print 'before tf.nn.softmax time:', time.ctime()
  prob       = tf.nn.softmax(tf.concat(tower_logits, 0), dim=1)
  print 'before tf.argmax time:', time.ctime()
  prob_class = tf.argmax(prob, axis=1)
  print 'before tf.metrics.accuracy time:', time.ctime()
  accuracy   = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=prob_class)
  ## train mode and eval mode ##
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    print 'before tf.estiamtor.EstimatorSpe time:', time.ctime()
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, eval_metric_ops={'accuracy':accuracy})

def main(unused_argv):
  run_config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto( \
                                                                    gpu_options          =tf.GPUOptions(allow_growth=True), \
                                                                    device_count         = {'GPU':4}, \
                                                                    allow_soft_placement = True))
  params= {}; params['gpu_num'] = 4; params['columns']=build_column()
  model = tf.estimator.Estimator( \
             model_fn = mode_mine, \
             params   = params, \
             config   = run_config, \
             model_dir= './model')
  for n in range(100//1):
    model.train(input_fn = lambda: input_fn('./iris.data', 1, True, 16))
    eval_res = model.evaluate(input_fn = lambda: input_fn('./iris.data', 1, False, 16))
    print 'epoch:', (n+1)*1, time.ctime()
    print '-'*40
    for key in sorted(eval_res):
      print key, eval_res[key]
if __name__ =='__main__':
  tf.app.run(main=main, argv=[sys.argv[0]]) 
