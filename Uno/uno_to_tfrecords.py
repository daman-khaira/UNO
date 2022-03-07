from __future__ import division, print_function

import logging
import os, sys, shutil

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
from tensorflow.python import ipu
from tensorflow.keras.models import Model

import uno as benchmark
# Import Candle libraries
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
import candle

import uno_data
from uno_data import CombinedDataLoader, CombinedDataGenerator, DataFeeder, TFDataFeeder
from model import build_model
from uno_baseline_keras2 import extension_from_parameters, initialize_parameters

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_feature(feature, label):
  #define the dictionary -- the structure -- of our single example
  data = {
        'feature'   : _bytes_feature(serialize_array(feature)),
        'label'     : _float_feature(label)
    }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out

def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'feature'   : tf.io.FixedLenFeature([], tf.string),
      'label' : tf.io.FixedLenFeature([], tf.float32),
    }

    
  content = tf.io.parse_single_example(element, data)
  
  feature_raw  = content['feature']
  feature = tf.io.parse_tensor(feature_raw, out_type=tf.float32)
  label = content['label']
  return (feature, label)

def write_feature_to_tfr_short(feature, labels, filename:str="features"):
  filename= filename+".tfrecords"
  writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
  count = 0

  rows, cols = feature.shape
  for index in range(rows):

    #get the data we want to write
    current_feat = feature[index,:] 
    current_label = labels[index]

    out = parse_single_feature( current_feat, current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count

def create_tfr_from_h5( args, logger=None ):
  for partition in ['train', 'val']:
    # Create tensorflow directory
    tfr_dir = os.path.join(args.export_data,partition)
    os.makedirs(tfr_dir)

    data_feeder = DataFeeder( partition= partition, filename=args.use_exported_data, batch_size=args.batch_size, shuffle=False, single=args.single, agg_dose=args.agg_dose, on_memory=args.on_memory_loader)
    
    for di in range(data_feeder.steps):
      tf_fname = os.path.join( tfr_dir, "data_%d"%(di) )

      feature, label = data_feeder[di]

      # combine the columns of all features into one matrix
      feature_vals = [fi.values for fi in feature]
      feature_mat = np.concatenate(feature_vals, axis=1)
      label       = label.values
      write_feature_to_tfr_short( feature_mat, label, filename=tf_fname )
      if logger is not None:
        logger.info('Generating {} dataset. {} / {}'.format(partition, di, data_feeder.steps))

def create_tfr_from_generator(args, logger=None):
  """ Not implemented yet: CombinedDataGenerator will be used to create tf records """
  return



def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))

    # Use exported data
    shutil.rmtree(args.export_data, ignore_errors=True)

    if args.use_exported_data is not None:
      create_tfr_from_h5(args, logger=logger)
    else:
      create_tfr_from_generator(args, logger=logger)
    
    return


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
