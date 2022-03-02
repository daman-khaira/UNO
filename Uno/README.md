### Training UNO model with IPU
This repo contains the code is a stripped down version of UNO project located at: https://github.com/ECP-CANDLE/Benchmarks
The code is adapted to use Graphcore's IPU hardware for UNO model's training and inference.

## Install Poplar SDK for IPU
Follow the instructions described [here](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#sdk-installation) to install the POPSDK for Tensorflow. This code uses tensorflow & keras framework which is also supported for the IPU. To learn more about using tensorflow for IPU, please refer to the [documentation](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/index.html) and available [tutorials](https://github.com/graphcore/tutorials/tree/master/tutorials/tensorflow2)

## Install Prerequisites
Once the correct python environment is activated as described in the previous section, install the required prequisites using
```
        pip install -r requirements.txt
```
## Run training on AUC dataset

First, download the AUC dataset using the following command:
```
wget -o top_21_auc_1fold.uno.h5 http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
```
To train the UNO model on the dataset downloaded previously, use:
```
  python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data data_dir/top_21_auc_1fold.uno.h5 -e 3 --save_weights save/saved.model.weights.h5
```
