import os
import sys

from tensorflow.python import pywrap_tensorflow

#model_dir='../model_lr'
#checkpoint_path = os.path.join(model_dir, "model.ckpt-308385")
checkpoint_path=sys.argv[1]
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key, reader.get_tensor(key).shape, reader.get_tensor(key))
    #if key.find('tbtag_embedding')>0: print reader.get_tensor(key)
	#print 'tensor_name:', key, var_to_shape_map[key]

