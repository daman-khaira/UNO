"""
TensorFlow record generator from ECP-CANDLE Uno data
"""

import argparse
import json
import os
from collections import OrderedDict
import six
import numpy as np
import tensorflow as tf
from ecp_candle.uno_data import CombinedDataLoader, CombinedDataGenerator

TRAIN_SOURCES = [
    'ALL', 'CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'SCL', 'SCLC', 'ALMANAC']


def get_arguments():
    """
    Creates arguments dictionary for Uno TFRecord generation from
    command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--in_dir',
        type=str,
        required=True,
        help='path to directory with raw Uno data; this argument is '
             'required')
    parser.add_argument(
        '--out_dir',
        default='./data_dir/',
        help='directory where TFRecords and data info will be stored; '
             'this directory will be expanded - default is ./data_dir/')
    parser.add_argument(
        '--fmt',
        default='fixed',
        choices=['segmented', 'fixed'],
        help='TFRecord dataset format; use fixed for compatibility with'
             'data APT in data.py - fixed by default')
    parser.add_argument(
        '--train_sources',
        nargs='+',
        choices=TRAIN_SOURCES,
        default=['ALL'],
        help='sources of drug response data for training; all by default')
    parser.add_argument(
        '--cache',
        default=None,
        help='prefix of data cache files to use, None otherwise; '
             'default is no cache file')
    parser.add_argument(
        '--seed',
        type=int,
        default=2018,
        help='seed for random number generation; default is 2018')
    parser.add_argument(
        '--use_landmark_genes',
        type=bool,
        default=True,
        help='whether to use the 978 landmark genes from LINCS (L1000) '
             'as expression features; defaults to True')
    parser.add_argument(
        '--preprocess_rnaseq',
        default='source_scale',
        choices=['source_scale', 'combat', 'none'],
        help='preprocessing method for RNAseq data; '
             'default is source_scale')
    parser.add_argument(
        '--single',
        action='store_true',
        help='pass this flag if you do not want to use drug pair '
             'representation')
    parser.add_argument(
        '--drug_median_response_min',
        type=float,
        default=-1.0,
        help='drugs whose median response is greater than this '
             'threshold will be kept; defaults to -1')
    parser.add_argument(
        '--drug_median_response_max',
        type=float,
        default=1.0,
        help='drugs with median response below this threshold '
             'will be kept; defaults to 1')
    parser.add_argument(
        '--use_filtered_genes',
        type=bool,
        default=False,
        help='whether to use the variance filtered genes as expression '
             'features; defaults to False')
    parser.add_argument(
        '--embed_features_source',
        type=bool,
        default=False,
        help='whether to embed the features source as part of the '
             'input; defaults to False')
    parser.add_argument(
        '--encode_response_source',
        type=bool,
        default=False,
        help='whether to encode the response data source as an '
             'input feature; defaults to False')
    parser.add_argument(
        '--test_sources',
        nargs='*',
        default=['train'],
        choices=['train'] + TRAIN_SOURCES,
        help='sources of drug response data for testing; '
             'defaults to same sources used for training')
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='proportion of data to use for validation; default is 0.2')
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help='whether to pre-shuffle examples before writing TF records; '
        'defaults to True')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2 ** 16,
        help='samples per batch in data generators; default is 2 ** 16')

    return parser.parse_args()


def order_train_sources(train_sources):
    """
    Sort and remove redundant train sources.
    Args:
        train_sources: (list of str)
            Training sources
    Returns:
        : (list of str)
            Ordered train sources
    """
    if 'ALL' in train_sources:
        return ['all']
    return sorted(set(train_sources))


def build_uno_data_loader(
        data_dir,
        train_sources,
        cache=None,
        seed=2018,
        cell_features=None,
        drug_features=None,
        drug_median_response_min=-1.0,
        drug_median_response_max=1.0,
        use_landmark_genes=True,
        use_filtered_genes=False,
        preprocess_rnaseq='source_scale',
        single=False,
        embed_feature_source=False,
        encode_response_source=False,
        test_sources=None):
    """
    Builds Uno Combined Data Loader and loads response data
    from the specified data sources.
    Args:
        data_dir: (str)
            Path to raw Uno data
        train_sources: (list of str)
            List of training sources.
        cache: (str, optional)
            Prefix of data cache files.
        seed: (int, optional)
            Seed for random number generation.
        cell_features: (str, optional)
            One of 'rnaseq' or 'none'; whether to use cell
            features as inputs or none at all.
            If None, 'rnaseq' will be used.
        drug_features: (list of str, optional)
            Which drug features to use as inputs; options
            are 'fingerprints', 'descriptors', and 'none'.
            If None, fingerprints and descriptors will be
            used,
        drug_median_response_min: (float, optional)
            Drugs whose median response is greater than
            this threshold will be kept.
        drug_median_response_max: (float, optional)
            Drugs whose median response is below this
            threshold will be kept.
        use_landmark_genes: (bool, optional)
            Whether to use the 978 landmark genes for LINCS
            as expression features.
        use_filtered_genes: (bool, optional)
            Whether to use the variance filtered genes as
            expression features.
        preprocess_rnaseq: (str, optional)
            Pre-processing method for cell_features.
        single: (bool, optional)
            Whether to use single drug representation.
        embed_feature_source: (bool, optional)
            Whether to embed the feature source as an
            input feature.
        encode_response_source: (bool, optional)
            Whether to encode the response data source as
            an input feature.
        test_sources: (list of str, optional)
            Sources of drug response data for testing.
            If None, same as training sources.
    Returns:
        : (CombinedDataLoader object)
            Loaded Uno drug response data.
    """
    print('Loading UNO data...')
    print('Training sources:', train_sources)

    if cell_features is None:
        cell_features = ['rnaseq']
    if drug_features is None:
        drug_features = ['descriptors', 'fingerprints']
    if test_sources is None:
        test_sources = ['train']

    loader = CombinedDataLoader(seed=seed, directory=data_dir)
    loader.load(
        cache=cache,
        ncols=0,
        cell_features=cell_features,
        drug_features=drug_features,
        drug_median_response_min=drug_median_response_min,
        drug_median_response_max=drug_median_response_max,
        use_landmark_genes=use_landmark_genes,
        use_filtered_genes=use_filtered_genes,
        preprocess_rnaseq=preprocess_rnaseq,
        single=single,
        train_sources=train_sources,
        test_sources=test_sources,
        embed_feature_source=embed_feature_source,
        encode_response_source=encode_response_source)

    return loader


def partition_data_loader(
        loader,
        val_split=0.2,
        cell_types=None,
        by_cell=None,
        by_drug=None,
        cv_folds=1):
    """
    Partitions loaded data into train and validation sets.
    Arguments:
        loader: (CombinedDataLoader object)
            Data loader with Uno drug response data.
        val_split: (float, optional)
            Proportion of data to use for validation.
        cell_types: (list of str, optional)
            Tissue types to limit data to if desired,
            None otherwise.
        by_cell: (str, optional)
            Cell ID if building a by-cell model,
            None otherwise.
        by_drug: (str, optional)
            Drug ID if building a by-drug model,
            None otherwise.
        cv_folds: (int, optional)
            Number of cross-validation folds.
    Returns:
        : (CombinedDataLoader object)
            Partitioned Uno drug response data.
    """
    train_split = 1 - val_split

    loader.partition_data(
        cv_folds=cv_folds,
        train_split=train_split,
        val_split=val_split,
        cell_types=cell_types,
        by_cell=by_cell,
        by_drug=by_drug)

    return loader


def extend_out_path(out_dir, train_sources, fmt, single=False):
    """
    Extends output path using information about the
    drug response sources and the format of the TF
    records.
    Arguments:
        out_dir: (str)
            Base output path.
        train_sources: (list of str)
            List of training sources
        fmt: (str)
            Format of TF records. One of fixed
            or segmented.
        single: (bool, optional)
            Whether the data uses single drug
            representation
    Returns:
        : (str)
        Extended output path.
    """
    subdir = '_'.join(train_sources)
    if single:
        subdir += '_single'
    out_dir_exp = os.path.join(out_dir, fmt, subdir)
    prefix = '_'.join(['uno', fmt, subdir])
    return out_dir_exp, prefix


def write_dataset_summaries(loader, out_dir, prefix):
    """
    Writes feature and example statistics of Uno dataset.
    Arguments:
        loader: (CombinedDataLoader)
            Partitioned data loader with Uno drug
            response data.
        out_dir: (str)
            Path to write metadata.
        prefix: (str)
            Prefix for summary files.
    Returns:
        : (tuple of OrderedDict)
            Feature and example summaries.
    """
    feature_names = list(loader.input_features.keys())
    feature_types = list(loader.input_features.values())

    data_width = 0
    input_sizes = []

    for ft_type in feature_types:
        ft_size, = loader.feature_shapes[ft_type]
        input_sizes.append(ft_size)
        data_width += ft_size

    assert data_width == loader.input_dim

    feat_metadata = OrderedDict()
    samp_smry = OrderedDict()

    for ft_name, ft_type, ft_shape in zip(
            feature_names, feature_types, input_sizes):
        feat_metadata[ft_name] = (ft_shape, 'float', ft_type)

    feat_metadata['label'] = (1, 'float', 'label')

    feat_json_name = _get_filename(
        out_dir, ('%s_features_smry' % prefix), 'json')
    with open(feat_json_name, "w") as json_file:
        json.dump(feat_metadata, json_file)

    train_examples = loader.train_indexes[0].size
    val_examples = loader.val_indexes[0].size
    total_examples = train_examples + val_examples

    samp_smry['total_examples'] = total_examples
    samp_smry['train_examples'] = train_examples
    samp_smry['val_examples'] = val_examples

    samp_json_name = _get_filename(
        out_dir, ('%s_examples_smry' % prefix), 'json')
    with open(samp_json_name, "w") as json_file:
        json.dump(samp_smry, json_file)

    feat_smry = {
        'feature_names': feature_names,
        'feature_types': feature_types,
        'input_sizes': input_sizes,
        'data_width': data_width,
        'metadata': feat_metadata}

    return (feat_smry, samp_smry)


def generate_tfrecords(
        loader,
        partition,
        out_dir,
        prefix,
        features_smry,
        single,
        batch_size=2 ** 16,
        shuffle=True,
        fmt='fixed',
        num_shards=1):
    """
    Builds Uno data generator with a partition of the
    loaded Uno data and loops through its batches to
    generate TF records.
    Arguments:
        loader: (CombinedDataLoader)
            Data loader with Uno data.
        partition: (str)
            One of 'train' or 'val'
        out_dir: (str)
            Path where TF records will be written.
        prefix: (str)
            Prefix for TF record filenames.
        features_smry: (OrderedDict)
            Dictionary with feature statistics
            corresponding to the loader's data.
        single: (bool)
            Whether the data loader uses single
            drug representation.
        batch_size: (int, optional)
            Number of examples in generator's mini
            batches.
        shuffle: (bool, optional)
            Whether to shuffle patitioned dataset
            before writing it into records.
        fmt: (str, pptional)
            One of 'fixed' or 'segmented'; type of
            records to generate.
        num_shards: (int, optional)
            Number of records to split each batch of
            data into.
    Returns:
        : (int)
            Number of examples written into records.
    """
    gen = CombinedDataGenerator(
        loader,
        partition=partition,
        batch_size=batch_size,
        shuffle=shuffle)

    total_rows = gen.size

    feature_names = features_smry["feature_names"]
    _, label_dtype, _ = features_smry["metadata"]["label"]

    print("\nProcessing {} set... ".format(partition))
    npasses = 0
    nrows = 0

    for features, labels in gen.flow(single=single):
        # Remove extra elements from batch, if on the last pass
        if npasses == np.ceil(total_rows / batch_size) - 1:
            last_batch_size = total_rows % batch_size
            features = [ft[:last_batch_size] for ft in features]
            labels = labels[:last_batch_size]

        features_list = [
            (name, ft) for name, ft in zip(feature_names, features)]

        name = ('%s_%s-%04d' % (prefix,  partition, npasses + 1))

        write_tf_records(
            features_list=features_list,
            labels=labels,
            out_dir=out_dir,
            name=name,
            fmt=fmt,
            label_type=label_dtype,
            num_shards=num_shards)

        npasses += 1
        nrows += len(labels)
        print(npasses, "-->", nrows, "total samples written")
        if nrows >= total_rows:
            break

    assert nrows == total_rows
    return nrows


def write_tf_records(
        features_list,
        labels,
        out_dir,
        name,
        fmt,
        label_type='float',
        num_shards=1):
    """
    Converts a set of features and labels into a TF records file.
    Arguments:
        features_list: (str, np.array)
            List of (name, feature) pairs with a batch of input
            features to write into TF records.
        labels: (np.array)
            Array of shape (num_examples, ) containing batch of
            labels to write into TF records.
        out_dir: (str)
            Directory where TF records will be stored.
        name: (str)
            Unqualified filename which may be extended by
            this function
        fmt: (str)
            TF records type, one of 'fixed' or 'segmented'.
        label_type: (str, optional)
            One of 'int' or 'float', representing type of
            labels.
        num_shards: (int, optional)
            Examples in batch are spread evenly among this number
            of files; default is 1.
    """
    ext = 'bin' if fmt == 'fixed' else 'tfrecords'
    if label_type == 'int':
        label_fn = _int64_feature
    elif label_type == 'float':
        label_fn = _float_feature
    else:
        assert False, "label_type value of 'int' or 'float' required"

    num_examples = labels.shape[0]

    def tf_record_writer(filename, start, end):
        if fmt == 'fixed':
            features = [ft for _, ft in features_list]
            return _fixed_tfrecord_writer(
                features, labels, filename, start, end)
        elif fmt == 'segmented':
            return _segmented_tfrecord_writer(
                features_list, labels, label_fn, filename, start, end)
        else:
            raise ValueError("fmt must be either 'fixed' or 'segmented'")

    if num_shards == 1:
        filename = _get_filename(out_dir, name, ext=ext)
        tf_record_writer(
            filename,
            start=0,
            end=num_examples)

    else:
        samples_per_batch = (num_examples + num_shards - 1) // num_shards
        print("writing {} files of {} records each".format(
            num_shards, samples_per_batch))

        for i in range(num_shards):
            start_index = samples_per_batch * i
            end_index = min(
                samples_per_batch + start_index, num_examples)
            filename = _get_filename(
                out_dir, '{}-{}'.format(name, i + 1), ext=ext)
            tf_record_writer(
                filename, start_index, end_index)

    return


def _fixed_tfrecord_writer(
        features,
        labels,
        filename,
        start=None,
        end=None,
        ft_dtype=np.float16,
        label_dtype=np.float16):
    """
    Writes a range of features and labels and writes data into a
    flat binary file.
    Arguments:
        features: (tensor)
        labels: (tensor)
            Set of labels
        filename: (str)
            Full path to TF record file
        start: (int)
            Start of range of rows of features and labels to slice.
        end: (int)
            End of range of rows of features and labels to slice.
        dtype: (type, type)
            Data types of features and labels.
    """
    data_width = np.sum([ft.shape[1] for ft in features])
    dtype = np.dtype([
        ('features', ft_dtype, data_width),
        ('label', label_dtype, 1)])

    data = []
    for i in range(start, end):
        ft_sliced = [ft[i, :] for ft in features]
        data.append((
            np.concatenate(ft_sliced),
            float(labels[i])))
    data = np.asarray(data, dtype=dtype)

    with open(filename, 'wb') as f:
        f.write(data.tobytes())

    print(filename)


def _segmented_tfrecord_writer(
        features_list, labels, label_fn, filename, start=None, end=None):
    """
    Serializes features and labels and writes data into a binary file in
    TFRecord format.
    Arguments:
        features_list: (list of (str, tensor))
            Set of named input features
        labels: (tensor)
            Set of labels
        filename: (str)
            Full path to TF record file
        start: (int)
            Start of range of rows of features and labels to slice
        end: (int)
            End of range of rows of features and labels to slice
    """
    with tf.python_io.TFRecordWriter(filename) as writer:
        for curr in range(start, end):
            feature_dict = {'label': label_fn(labels[curr])}

            for name, feature in features_list:
                byterepr = feature[curr].tostring()
                feature_dict[name] = _bytes_feature(byterepr)

            tf_features = tf.train.Features(feature=feature_dict)
            example = tf.train.Example(features=tf_features)
            writer.write(example.SerializeToString())

    print(filename)


def _get_filename(data_dir, name, ext='tfrecords'):
    """
    Construct a full path to a file to be stored in the
    data_dir. Will also ensure the data directory exists.
    Args:
        data_dir: (str)
            The directory where the file will be stored
        name: (str)
            The unqualified file name
        ext: (str)
            The filename's extension
    Returns:
        : (str)
            The full path to the TFRecord file
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    return os.path.join(data_dir, '{}.{}'.format(name, ext))


def _int64_feature(value):
    """
    Creates tf.Train.Feature from int64 value.
    """
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """
    Creates tf.Train.Feature from float value.
    """
    if value is None:
        value = []
    if isinstance(value, np.ndarray) and value.ndim > 1:
        value = value.reshape(-1)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
    Creates tf.Train.Feature from bytes value.
    """
    if value is None:
        value = []
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main():
    """
    Generates TF records from ECP_CANDLE Uno data.
    """
    args = get_arguments()

    in_dir = args.in_dir
    if not in_dir.endswith('/'):
        in_dir += '/'
    train_sources = order_train_sources(args.train_sources)
    out_dir = args.out_dir

    loader = build_uno_data_loader(
        data_dir=in_dir,
        train_sources=train_sources,
        cache=args.cache,
        seed=args.seed,
        drug_median_response_min=args.drug_median_response_min,
        drug_median_response_max=args.drug_median_response_max,
        use_landmark_genes=args.use_landmark_genes,
        use_filtered_genes=args.use_filtered_genes,
        preprocess_rnaseq=args.preprocess_rnaseq,
        single=args.single,
        embed_feature_source=args.embed_features_source,
        encode_response_source=args.encode_response_source,
        test_sources=args.test_sources)

    loader = partition_data_loader(loader, args.val_split)

    out_dir_exp, prefix = extend_out_path(
        out_dir, train_sources, args.fmt, args.single)

    feat_smry, ex_smry = write_dataset_summaries(
        loader, out_dir_exp, prefix)

    feature_names = feat_smry["feature_names"]
    input_sizes = feat_smry["input_sizes"]
    print('Feature keys in the generated training and '
          'validation datasets will be:')
    for name, size in zip(feature_names, input_sizes):
        print("\t {} with associated shape ({},)".format(
            name, size))

    for partition in ['train', 'val']:
        n_examples = generate_tfrecords(
            loader,
            partition,
            out_dir_exp,
            prefix,
            feat_smry,
            batch_size=args.batch_size,
            single=args.single,
            fmt=args.fmt)
        assert n_examples == ex_smry["{}_examples".format(partition)]

    print('\n%s TFRecords written to %s.\n' % (args.fmt, out_dir_exp))


if __name__ == '__main__':
    main()
