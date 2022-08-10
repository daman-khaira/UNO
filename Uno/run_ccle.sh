#!/bin/bash

bsz=512
spe=-1 # Steps per exec
epochs=10
IOTiles=128
dropout=0.3
data_type='f16'

#python uno_baseline_keras2.py --train_sources CCLE --cache ./data_dir/cache/CCLE_fp16 --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --export_data ./data_dir/CCLE_fp16.h5 --export_tfrecords CCLE_TFR_FP16 --cp True --shuffle True --on_memory_loader False
python uno_baseline_keras2.py --train_sources CCLE --cache ./data_dir/cache/CCLE_fp16 --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --use_exported_data ./data_dir/CCLE_fp16.h5 --shuffle True --on_memory_loader True --steps_per_execution $spe --epochs 20
#python uno_baseline_keras2.py --train_sources CCLE --cache ./data_dir/cache --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --use_exported_data ./data_dir/CCLE.h5 --use_tfrecords ./data_dir/CCLE_TFR --cp True --shuffle True --on_memory_loader False --steps_per_execution $spe --epochs 20
#python uno_baseline_keras2.py --train_sources CCLE --cache ./data_dir/cache/CCLE --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $bsz --use_exported_data ./data_dir/CCLE.h5 --cp True --shuffle True --on_memory_loader False --steps_per_execution $spe --epochs $epochs --num_io_tiles $IOTiles --use_tfrecords CCLE_TFR --dropout $dropout --data_type $data_type --learning_rate 0.0001
