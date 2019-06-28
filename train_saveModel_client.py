#!/root/anaconda3/bin/python
from __future__ import print_function

import sys 
import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from DataRead import DataReadAndNegSamp, getNow 


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 10000, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """

  #test_data_set = mnist_input_data.read_data_sets(work_dir).test
  test_data_set = DataReadAndNegSamp(file_input='./20190623.ID.test')
  test_data_set = test_data_set.train_data.values
  print ('test_data_set.shape:', test_data_set.shape, type(test_data_set))
  channel  = grpc.insecure_channel(hostport)
  stub     = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  pred_res = []
  real_res = []
  for index in range(int(num_tests/concurrency)):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'seq_model'
    request.model_spec.signature_name = 'pred_class'
    #request.model_spec.signature_name = 'pred_prob' ## here used 'input'
    cur_data    = test_data_set[index*concurrency:(index+1)*concurrency]
    cur_label   = cur_data[:, 0]
    cur_feature = cur_data[:, 1:].reshape((-1, 13))
    #print ('cur_data.shape:', cur_data.shape, type(cur_data))
    #print ('cur_label.shape:', cur_label.shape, type(cur_label))
    #print ('cur_feature.shape:', cur_feature.shape, type(cur_feature))
    #image, label = test_data_set[index]
    #request.inputs['input'].CopyFrom(
    request.inputs['input_'].CopyFrom(
        tf.contrib.util.make_tensor_proto(values=cur_feature, dtype=tf.int32, shape=cur_feature.shape))
        #tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
    response = stub.Predict(request, 5.0)
    #print ('respose:', response, type(response))
    results  = tf.contrib.util.make_ndarray(response.outputs['output_'])
    pred_res.append(list(results))
    real_res.append(list(cur_label))
    '''
    ## here means saveModel can multi-ouput sig_def ##
    results  = {}
    for key in response.outputs:
      tensor_proto = response.outputs[key]
      nd_array = tf.contrib.util.make_ndarray(tensor_proto)
      results[key] = nd_array
    for key, values in results.items():
      print ('in result:', key, values)
    '''
  pred_res = np.array(pred_res).reshape((-1,1))
  real_res = np.array(pred_res).reshape((-1,1))
  res = np.concatenate((real_res, pred_res), axis=1)
  print ('real-label\t pred-label \n', res)
  return np.sum(np.equal(res[:,0], res[:, 1]))/res.shape[0]


def main(_):
  if FLAGS.num_tests > 10001:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests)
  print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()
