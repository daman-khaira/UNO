#!/bin/bash
IOTiles=128

python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data data_dir/top_21_auc_1fold.uno.h5 -e 10 --save_weights save/saved.model.weights.h5 --steps_per_execution -1 --num_io_tiles $IOTiles
