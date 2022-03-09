from __future__ import division, print_function

import logging
import os, sys, shutil

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


import uno as benchmark
# Import Candle libraries
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
import candle

from uno_data import DataFeeder, CombinedDataGenerator
from uno_tfr_utils import *
from uno_baseline_keras2 import extension_from_parameters, initialize_parameters

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
