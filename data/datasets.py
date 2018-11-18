import tensorflow as tf
from data.preprocessing import make_preprocessing_pipeline


def create_dataset(data_config, numclasses, preprocessing_config, mode):
    data_type = data_config['data_type']
    if data_type not in ('tfrecord', 'image_list'):
        raise ValueError('Only "tfrecord" and "image_list" are supported as '
                         'valid input data formats.')

    if not isinstance(numclasses, int):
        raise ValueError('numclasses must be a positive integer.')

    if numclasses <= 0:
        raise ValueError(
            'numclasses must be a positive integer. It is {}'.format(
                numclasses))

    elif data_type == 'tfrecord':
        tfrecord_options = data_config['options']
        return tfrecord_db(tfrecord_options, preprocessing_config,
                           numclasses, mode)
    else:
        image_list_options = data_config['options']
        return imagelist_db(image_list_options, preprocessing_config)


def tfrecord_db(tfrecord_options, preprocessing_config, numclasses,
                mode):
    tfrecord_pattern = tfrecord_options['tfrecord_pattern']
    if not isinstance(tfrecord_pattern, str):
        raise ValueError('tfrecord_pattern must be a string.')

    if len(tfrecord_pattern) == 0:
        raise ValueError('No tfrecord_pattern was specified.')

    num_parallel_reads = tfrecord_options['num_parallel_reads']
    if not isinstance(num_parallel_reads, int):
        raise ValueError('num_parallel_reads must be an integer.')

    if mode not in ('train', 'eval'):
        raise ValueError('TFRecords can be used only during training or '
                         'evaluation. For prediction use image_list.')

    if num_parallel_reads < 0:
        num_parallel_reads = None
        tf.logging.info('num_parallel_reads was a negative integer. Setting '
                        'it to None. The records will be read sequentially.')

    read_buffer_size = tfrecord_options['read_buffer_size']

    if not isinstance(read_buffer_size, int):
        raise ValueError('buffer_size for reading TFRecords must be a '
                         'positive integer.')

    if read_buffer_size < 0:
        read_buffer_size = None
        tf.logging.info('read_buffer_size was a negative number. Setting it '
                        'to None. This will set it to 256 KB by tensorflow.')

    tfrecord_files = tf.gfile.Glob(tfrecord_pattern)
    if len(tfrecord_files) == 0:
        raise ValueError('No tfrecord files matching the pattern "{}" were '
                         'found.'.format(tfrecord_pattern))

    tf.logging.info('tfrecord_files found : {}'.format(len(tfrecord_files)))

    db = tf.data.TFRecordDataset(filenames=tfrecord_files,
                                 buffer_size=read_buffer_size,
                                 num_parallel_reads=num_parallel_reads)

    shuffle = tfrecord_options['shuffle']
    if not isinstance(shuffle, bool):
        raise ValueError('shuffle parameter must be a boolean.')

    if shuffle:
        shuffle_buffer_size = tfrecord_options['shuffle_buffer_size']
        if not isinstance(shuffle_buffer_size, int) or shuffle_buffer_size <= 0:
            raise ValueError('shuffle_buffer_size must be a positive integer')
        reshuffle_each_iteration = tfrecord_options['reshuffle_each_iteration']
        if not isinstance(reshuffle_each_iteration, bool):
            raise ValueError('reshuffle_each_iteration must be a boolean.')

        db = db.shuffle(buffer_size=shuffle_buffer_size,
                        reshuffle_each_iteration=reshuffle_each_iteration)

    cache = tfrecord_options['cache']
    if not isinstance(cache, bool):
        raise ValueError('cache parameter must be a boolean.')

    if cache:
        cache_dir = tfrecord_options['cache_dir']
        if not isinstance(cache_dir, str):
            raise ValueError('cache_dir must be a string denoting '
                             'path to cache the dataset.')

        if len(cache_dir) == 0:
            tf.logging.info('cache_dir was set to None. This means that the '
                            'dataset will be cached in memory. If you do not '
                            'have enough memory, it could lead to a crash.')
        elif not tf.gfile.IsDirectory(cache_dir):
            raise ValueError(
                'cache_dir must be None or a string denoting a valid directory')
        else:
            if not tf.gfile.Exists(cache_dir):
                tf.gfile.MkDir(cache_dir)

        db = db.cache(cache_dir)

    preprocessing_num_parallel_calls = tfrecord_options[
        'preprocessing_num_parallel_calls']
    if not isinstance(preprocessing_num_parallel_calls, int) and not None:
        raise ValueError(
            'preprocessing_num_parallel_calls must be a positive integer or None.')

    if preprocessing_num_parallel_calls <= 0:
        preprocessing_num_parallel_calls = None
        tf.logging.warn(
            'preprocessing_num_parallel calles was specified as a non-positive integer. It is hence set to None. This will mean that '
            'preprocessing will happen sequentially. This can '
            'affect speed.')

    db = db.map(lambda x: make_preprocessing_pipeline(x,
                                                      preprocessing_config,
                                                      numclasses,
                                                      mode,
                                                      'tfrecord'),
                num_parallel_calls=preprocessing_num_parallel_calls)

    batchsize = tfrecord_options['batchsize']
    if not isinstance(batchsize, int):
        raise ValueError('batchsize must be an integer.')

    if batchsize <= 0:
        raise ValueError('batchsize must be an integer greater than 0.')

    drop_remainder = tfrecord_options['drop_remainder']
    if not isinstance(drop_remainder, bool):
        raise ValueError('drop_remainder must be a bool.')

    db = db.batch(batch_size=batchsize, drop_remainder=drop_remainder)

    prefetch_buffer_size = tfrecord_options['prefetch_buffer_size']
    if not isinstance(prefetch_buffer_size, int):
        raise ValueError(
            'prefetch_buffer_size must be an integer and greater than zero.')

    if prefetch_buffer_size <= 0:
        raise ValueError('prefetch_buffer_size must be an integer greater '
                         'than zero.')

    db = db.prefetch(buffer_size=prefetch_buffer_size)

    return db


def imagelist_db(image_list_options, preprocessing_config):
    image_list_file = image_list_options['image_list_file']
    if not isinstance(image_list_file, str):
        raise ValueError('image_list_file must be a string.')

    if len(image_list_file) == 0:
        raise ValueError('image_list_file cannot be an empty string.')

    if not tf.gfile.Exists(image_list_file):
        raise ValueError('The file {} does not exist.'.format(image_list_file))

    num_header_lines = image_list_options['num_header_lines']
    if not isinstance(num_header_lines, int):
        raise ValueError('num_header_lines must be an integer.')

    if num_header_lines <= 0:
        num_header_lines = 0
        tf.logging.warn('num_header_lines was a negative integer. Setting it '
                        'to 0.')

    read_buffer_size = image_list_options['read_buffer_size']

    if not isinstance(read_buffer_size, int):
        raise ValueError('buffer_size for reading image_list_files must be a '
                         'positive integer.')

    if read_buffer_size < 0:
        read_buffer_size = None
        tf.logging.info('read_buffer_size was a negative number. Setting it '
                        'to None. This will set it to 256 KB by tensorflow.')

    db = tf.data.TextLineDataset(image_list_file, buffer_size=read_buffer_size)
    if num_header_lines > 0:
        db = db.skip(num_header_lines)

    shuffle = image_list_options['shuffle']
    if not isinstance(shuffle, bool):
        raise ValueError('shuffle parameter must be a boolean.')
    if shuffle:
        shuffle_buffer_size = image_list_options['shuffle_buffer_size']
        if not isinstance(shuffle_buffer_size, int) or shuffle_buffer_size <= 0:
            raise ValueError('shuffle_buffer_size must be a positive integer')
        reshuffle_each_iteration = image_list_options[
            'reshuffle_each_iteration']
        if not isinstance(reshuffle_each_iteration, bool):
            raise ValueError('reshuffle_each_iteration must be a boolean.')

        db = db.shuffle(buffer_size=shuffle_buffer_size,
                        reshuffle_each_iteration=reshuffle_each_iteration)

    cache = image_list_options['cache']
    if not isinstance(cache, bool):
        raise ValueError('cache parameter must be a boolean.')

    if cache:
        cache_dir = image_list_options['cache_dir']
        if not isinstance(cache_dir, str) and not None:
            raise ValueError('cache_dir must be None or a string denoting '
                             'path to cache the dataset.')

        if cache_dir is not None:
            if not tf.gfile.IsDirectory(cache_dir):
                raise ValueError(
                    'cache_dir must be None or a string denoting a valid director')

            if not tf.gfile.Exists(cache_dir):
                tf.gfile.MkDir(cache_dir)
        else:
            tf.logging.info('cache_dir was set to None. This means that the '
                            'dataset will be cached in memory. If you do not '
                            'have enough memory, it could lead to a crash.')

            db = db.cache(cache_dir)

    preprocessing_num_parallel_calls = image_list_options[
        'preprocessing_num_parallel_calls']
    if not isinstance(preprocessing_num_parallel_calls, int) and not None:
        raise ValueError(
            'preprocessing_num_parallel_calls must be a positive integer or None.')

    if preprocessing_num_parallel_calls <= 0:
        preprocessing_num_parallel_calls = None
        tf.logging.warn(
            'preprocessing_num_parallel calles was specified as a non-positive integer. It is hence set to None. This will mean that '
            'preprocessing will happen sequentially. This can '
            'affect speed.')

    db = db.map(lambda x: make_preprocessing_pipeline(x,
                                                      preprocessing_config,
                                                      'image_list'),
                num_parallel_calls=preprocessing_num_parallel_calls)

    batchsize = image_list_options['batchsize']
    if not isinstance(batchsize, int):
        raise ValueError('batchsize must be an integer.')

    if batchsize <= 0:
        raise ValueError('batchsize must be an integer greater than 0.')

    drop_remainder = image_list_options['drop_remainder']
    if not isinstance(drop_remainder, bool):
        raise ValueError('drop_remainder must be a bool.')

    db = db.batch(batchsize, drop_remainder=drop_remainder)

    return db
