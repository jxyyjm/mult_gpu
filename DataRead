#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
  1) read all positive samples
  2) neg sampling 
  3) merge and shuffle
'''

import os
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy  as np
import time
from sklearn.utils import shuffle
def getNow():
  return '['+str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(time.time()))))+']'

class DataReadAndNegSamp:
  def __init__(self, file_input='./'):
    self.file_input = file_input
    self.data_read  = self.pdreaddata()
    self.url_counts = self.freqcount()
    self.url_accuProb = self.freqprob()
    self.url_only_ids = self.urlonlyids()
    self.negative_samp= self.negsample()
    self.positive_samp= self.possample()
    self.train_data   = self.mergedata()
  def pdreaddata(self):
    data_read  = pd.read_table(self.file_input, sep='\t', dtype=np.int32, \
         names = ['user', 'url', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
    print 'read data :', getNow()
    return data_read
  def freqcount(self):
    url_counts = pd.value_counts(self.data_read['url']) ## 统计频次，且自动从大到小排序 ##
    return url_counts
  def freqprob(self):
    url_freqSum  = np.sum(self.url_counts.values)       ## 计算总频次 ## 
    url_accuProb = self.url_counts.divide(url_freqSum)  ## 计算url的频率比例 ##
    return url_accuProb
    #print 'urls count: ', getNow()
  def urlonlyids(self):
    url_only_ids = pd.DataFrame(self.data_read, \
         columns = ['url', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
    url_only_ids = url_only_ids.drop_duplicates('url') ## url去重
    return url_only_ids
  def urlsample(self):
    url_sample   = self.url_only_ids.sample( \
         n       = self.data_read.shape[0], \
         replace = True, \
         weights = self.url_accuProb)
    return url_sample
  def negsample(self):
    url_sample   = self.urlsample()
    negative_samp= np.concatenate((self.data_read['user'].to_frame(), url_sample), axis=1)
    negative_labe= np.zeros(shape=(self.data_read.shape[0],1), dtype=np.int32)
    negative_samp= np.concatenate((negative_labe, negative_samp), axis=1)
    print 'neg sample:', getNow()
    return negative_samp
  def possample(self):
    positive_labe= np.ones(shape=(self.data_read.shape[0], 1), dtype=np.int32)
    positive_samp= np.concatenate((positive_labe, self.data_read), axis=1)
    return positive_samp
  def mergedata(self):
    train_data   = np.concatenate((self.positive_samp, self.negative_samp), axis=0)
    train_data   = pd.DataFrame(data = train_data, \
                   columns = ['label', 'user', 'url', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
    train_data   = shuffle(train_data)
    print 'all train :', getNow()
    return train_data

if __name__=='__main__':
  dataclass = DataNegSamp(file_input='./user_click_urls.ID')
  print 'input.shape:', dataclass.data_read.shape
  print 'train.shape:', dataclass.train_data.shape
  
