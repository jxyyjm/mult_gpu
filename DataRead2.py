#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
  1) read all positive samples
  2) neg sampling, again neg smpling support also
  3) merge and shuffle
  4) url-none-id random replace
  5) read extended negative samples
  notice: pd.DataFrame.sample, weight will align w.index to object.index 
'''

import os
import sys 
#reload(sys)
#sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy  as np
import time
from sklearn.utils import shuffle
def getNow():
  return '['+str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(time.time()))))+']'

class DataReadAndNegSamp:
  def __init__(self, file_input='./', seq_len=12, file_negative='./'):
    self.file_input = file_input
    self.file_neg   = file_negative
    self.seq_len    = seq_len
    self.col_names  = self.getcolnames()
    self.neg_samples= self.read_negative_samples()
    self.data_read  = self.pdreaddata()
    self.none_id    = self.get_none_index()
    self.url_counts = self.freqcount()
    self.url_accuProb = self.freqprob()
    self.url_only_ids = self.urlonlyids()
    self.negative_samp= self.negsample()
    self.positive_samp= self.possample()
    self.train_data   = self.mergedata()


  def getcolnames(self):
    return ['user'] +['seq'+str(i) for i in range(self.seq_len)]
  def pdreaddata(self):
    data_read  = pd.read_table(self.file_input, sep='\t', dtype=np.int32, names = self.col_names)
    print ('read data :', getNow())
    return data_read
  def freqcount(self):
    url_counts = pd.value_counts(self.data_read['seq0']) ## 统计频次，且自动从大到小排序 ##
    ## 大体上，seq1的统计是差不多的 ##
    return url_counts
  def freqprob(self):
    url_freqSum  = np.sum(self.url_counts.values)       ## 计算总频次 ## 
    url_accuProb = self.url_counts.divide(url_freqSum)  ## 计算url的频率比例 ##
    return url_accuProb
    #print ('urls count: ', getNow())
  def urlonlyids(self):
    url_only_ids = pd.DataFrame(self.data_read, columns = ['seq0'])
    url_only_ids = url_only_ids.drop_duplicates('seq0') ## url去重
    url_only_ids.index = url_only_ids['seq0'].values ## 将索引值设定为 url-ID ##
    return url_only_ids
  def urlsample(self, by_freq=True):
    if by_freq: weights = self.url_accuProb
    else:       weights = None
    url_sample   = self.url_only_ids.sample( \
         n       = self.data_read.shape[0]*self.seq_len, \
         replace = True, \
         weights = weights)
    url_sample = url_sample.values.reshape([self.data_read.shape[0], self.seq_len])
    return url_sample
  def negsample(self, by_freq=True, num=1):
    ## 1) user, sample, seq11, 0 ##
    ## 2) user, seq0~10, samp, 0 ##    
    url_sample   = self.urlsample(by_freq=by_freq)
    seq_samp_top = url_sample[:, 0:-1]
    seq_samp_tail= url_sample[:, -1:self.seq_len]
    user_seq_real= self.data_read['user'].to_frame()

    seq_real_tail       = self.data_read['seq'+str(self.seq_len-1)].to_frame()
    negative_samp_part1 = np.concatenate((user_seq_real, seq_samp_top, seq_real_tail), axis=1)
    negative_labe_part1 = np.zeros(shape=(self.data_read.shape[0],1), dtype=np.int32)
    negative_samp_part1 = np.concatenate((negative_labe_part1, negative_samp_part1), axis=1)

    seq_real_top        = self.data_read[['seq'+str(i) for i in range(self.seq_len-1)]]
    negative_samp_part2 = np.concatenate((user_seq_real, seq_real_top, seq_samp_tail), axis=1)
    negative_labe_part2 = np.zeros(shape=(self.data_read.shape[0],1), dtype=np.int32)
    negative_samp_part2 = np.concatenate((negative_labe_part2, negative_samp_part2), axis=1)

    print ('neg sample:', getNow())
    return np.concatenate((negative_samp_part1, negative_samp_part2), axis=0)

  def possample(self):
    positive_labe= np.ones(shape=(self.data_read.shape[0], 1), dtype=np.int32)
    positive_samp= np.concatenate((positive_labe, self.data_read), axis=1)
    return positive_samp
  def mergedata(self):
    if self.neg_samples.any() or self.neg_samples!=None:
      train_data = np.concatenate((self.positive_samp, self.negative_samp, self.neg_samples), axis=0)
    else:
      train_data = np.concatenate((self.positive_samp, self.negative_samp), axis=0)

    train_data   = pd.DataFrame(data = train_data, columns = ['label'] + self.col_names)
    train_data   = shuffle(train_data)
    print ('all train :', getNow())
    return train_data
  '''
  def mergedata(self):
    ## only use negative_samples real and positive_samples real
    if self.neg_samples.any() or self.neg_samples!=None:
      train_data = np.concatenate((self.positive_samp, self.neg_samples), axis=0)
    else:
      train_data = self.positive_samp
    train_data   = pd.DataFrame(data = train_data, columns = ['label'] + self.col_names)
    train_data   = shuffle(train_data)  
    print ('all train :', getNow())
    return train_data
  '''
  def samplingagain(self):
    self.negative_samp= self.negsample(by_freq=True, num=1)
    self.positive_samp= self.possample()
    self.train_data   = self.mergedata()
    #self.url_replace_random_none_id()
    return self.train_data

  def get_none_index(self):
    print ('this data has no none-index')
    return None
    #column_c = pd.value_counts(self.data_read['tag5'])
    #none_id  = column_c[0:1].index[0]
    #print ('get none-index:', none_id)
    #return none_id
  def url_replace_random_none_id(self):
    random_pos_value = self.train_data['url'].sample(frac=0.1)
    self.train_data['url'][random_pos_value.index] = self.none_id
    ## test which is faster ## index is faster than pd.replace ##
  def read_negative_samples(self):
    if os.path.isfile(self.file_neg)==False:
      print 'no neg-file named:', self.file_neg, 'load negative samples 0 to used'
      return None
    data_read = pd.read_table(self.file_neg, sep='\t', dtype=np.int32, names = self.col_names)
    neg_label = np.zeros(shape=(data_read.shape[0],1), dtype=np.int32)
    neg_samp  = np.concatenate((neg_label, data_read), axis=1)
    print 'load negatvie samples [ shape, with label ]:', neg_samp.shape, 'to used'
    return neg_samp

if __name__=='__main__':
  dataclass = DataReadAndNegSamp(file_input='../data/20190623.ID', seq_len=12)
  print ('input.shape:', dataclass.data_read.shape)
  print ('train.shape:', dataclass.train_data.shape)
