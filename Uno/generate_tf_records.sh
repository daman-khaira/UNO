#!/bin/bash

#python uno_to_tfrecords.py --train_sources all --cache data_dir/cache --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 4096 --use_exported_data data_dir/all.h5 --export_data ALL_TFR --on_memory_loader False
python uno_to_tfrecords.py --train_sources CCLE --cache data_dir/cache --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 4096 --use_exported_data data_dir/CCLE.h5 --export_data CCLE_TFR --on_memory_loader False
