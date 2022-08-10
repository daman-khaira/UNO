#!/bin/bash

bsz=512
spe=-1 # Steps per exec
tfr_dir='/localdata/damank/FE/uno/Uno/all_TFR_fp16'

# python uno_baseline_keras2.py --train_sources all --cache ./data_dir/cache --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --export_data ./data_dir/all_fp16.h5 --export_tfrecords all_TFR_fp16 --cp True --shuffle True --on_memory_loader False
#python uno_baseline_keras2.py --train_sources all --cache ./data_dir/cache --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --export_data ./data_dir/all_fp16.h5 --cp True --shuffle True --on_memory_loader False
#python uno_baseline_keras2.py --train_sources all --cache ./data_dir/cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --use_exported_data ./data_dir/all.h5 --use_tfrecords $tfr_dir --cp True --shuffle True --steps_per_execution $spe --epochs 5  --timeout -1 --dropout 0.3
python uno_baseline_keras2.py --train_sources all --cache ./data_dir/cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --use_exported_data ./data_dir/all_fp16.h5 --use_tfrecords $tfr_dir --cp True --shuffle True --steps_per_execution $spe --epochs 5  --timeout -1 --dropout 0.3
