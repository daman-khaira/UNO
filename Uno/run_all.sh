#!/bin/bash

bsz=256
spe=-1 # Steps per exec

#python uno_baseline_keras2.py --train_sources all --cache ./data_dir/cache --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --export_data ./data_dir/all.h5 --export_tfrecords all_TFR --cp True --shuffle True --on_memory_loader False
python uno_baseline_keras2.py --train_sources all --cache ./data_dir/cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --use_exported_data ./data_dir/all.h5 --use_tfrecords data_dir/ALL_TFR --cp True --shuffle True --steps_per_execution $spe --epochs 20  --timeout -1 --dropout 0.3
