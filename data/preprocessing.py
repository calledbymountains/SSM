import tensorflow as tf


def make_preprocessing_pipeline(data, preprocessing_config, numclasses, mode,
                                data_type):
    input_dims = dict(
        input_height=preprocessing_config['input_height'],
        input_width=preprocessing_config['input_width'],
        input_channels=3
    )

    for key, val in input_dims.items():
        if not isinstance(val, int):
            raise ValueError('{} is not an integer. It must be a positive '
                             'integer.')

        if val <= 0:
            raise ValueError('{} is not a positive integer. It must be a '
                             'positive integer.')

    if data_type not in ('tfrecord', 'image_list'):
        raise ValueError('data_type must be one of "tfrecord" and '
                         '"image_list".')

    if data_type == 'tfrecord':
        parsed_data = tfrecord_parser(data, preprocessing_config, numclasses,
                                      mode)
    else:
        parsed_data = imagelist_parser(data, preprocessing_config, numclasses)

    return parsed_data


def tfrecord_parser(record_proto, preprocessing_config, numclasses, mode):
    feature_map = {
        'image/height': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/colorspace': tf.FixedLenFeature([], tf.string,
                                               default_value='RGB'),
        'image/channels': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='JPEG'),
        'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/id': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/extra': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
        # 'image/class/text': tf.FixedLenFeature([], tf.string,
        # default_value=''),
        'image/class/conf': tf.FixedLenFeature([], tf.float32,
                                               default_value=0),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/text': tf.VarLenFeature(dtype=tf.string),
        'image/object/bbox/conf': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/score': tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/x': tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/y': tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/v': tf.VarLenFeature(dtype=tf.int64),
        'image/object/parts/score': tf.VarLenFeature(dtype=tf.float32),
        'image/object/count': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/object/area': tf.VarLenFeature(dtype=tf.float32),
        'image/object/id': tf.VarLenFeature(dtype=tf.string)
    }

    features = tf.parse_single_example(record_proto, features=feature_map)
    features['image/encoded'] = tf.image.decode_jpeg(features['image/encoded'],
                                                     channels=3)

    features['image/encoded'] = tf.reshape(features['image/encoded'],
                                           [features['image/height'],
                                            features['image/width'],
                                            features['image/channels']])

    features, labels = preprocessing_pipeline(features, preprocessing_config,
                                              mode,
                                              numclasses)

    return dict(
        features=features,
        labels=labels
    )


def imagelist_parser(text_data, preprocessing_config, numclasses):
    line_split = tf.string_split(text_data, delimiter='\t')
    image_file_name = line_split.values[0]
    image_height = tf.cast(line_split.values[1], tf.int64)
    image_width = tf.cast(line_split.values[2], tf.int64)
    features = dict()
    features['image/height'] = image_height
    features['image/width'] = image_width
    features['image/channels'] = 3
    features['image/encoded'] = tf.image.decode_jpeg(image_file_name,
                                                     channels=3)
    features['image/encoded'] = tf.reshape(
        features['image/encoded'], [image_height, image_width, 3])

    features = preprocessing_pipeline(features, preprocessing_config,
                                      mode='pred')

    return dict(
        features=features
    )


def map_new_range(image):
    with tf.variable_scope('map_range'):
        image /= 255.0
        image *= 2.0
        image -= 1.0
    return image


def preprocessing_pipeline(features, preprocessing_config, mode,
                           numclasses=None):
    max_num_boxes = preprocessing_config['max_num_boxes']
    if not isinstance(max_num_boxes, int) or max_num_boxes <= 0:
        raise ValueError('max_num_boxes must be a positive integer.')

    input_height = preprocessing_config['input_height']
    input_width = preprocessing_config['input_width']
    with tf.variable_scope('resize_to_input_size'):
        features['image/encoded'] = tf.image.resize_bilinear(
            tf.expand_dims(features[
                               'image/encoded'], 0),
            [input_height,
             input_width])
        features['image/encoded'] = tf.to_float(features['image/encoded'])
    features['image/encoded'] = map_new_range(features['image/encoded'])
    if mode == 'pred':
        features['image/encoded'] = tf.squeeze(features['image/encoded'],
                                               axis=0)
        features['image/encoded'].set_shape((input_height, input_width, 3))
        return features['image/encoded']

    bbox_xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'],
                                          default_value=0)
    bbox_xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'],
                                          default_value=0)
    bbox_ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'],
                                          default_value=0)
    bbox_ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'],
                                          default_value=0)
    with tf.variable_scope('bbox'):
        bboxes = tf.stack([bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax], axis=1)

    labels = tf.sparse_tensor_to_dense(features['image/object/bbox/label'],
                                       default_value=0)
    labels = tf.one_hot(labels, depth=numclasses)

    if mode == 'train':
        if 'random_horizontal_flip' in preprocessing_config[
            'data_augmentation'].keys():
            features['image/encoded'], labels = flip_image_and_bounding_boxes(
                features['image/encoded'], bboxes)
            tf.logging.info(
                'Random_horizontal_flipping of the image and bounding boxes will be applied randomly.')
        else:
            tf.logging.info('No random_horizontal_flipping of the image and '
                            'bounding boxes will be applied.')

        if not isinstance(preprocessing_config['data_augmentation'][
                              'random_brightness']['apply'],
                          bool):
            raise ValueError('apply for random_brightness must be a bool.')

        random_brightness_apply = \
        preprocessing_config['data_augmentation']['random_brightness'][
            'apply']
        if random_brightness_apply:
            options = preprocessing_config['data_augmentation'][
                'random_brightness']
            features['image/encoded'] = random_brightness(features[
                                                              'image/encoded'],
                                                          options)
        else:
            tf.logging.info('No random_brightness applied.')

        if not isinstance(
                preprocessing_config['data_augmentation']['random_contrast'][
                    'apply'],
                bool):
            raise ValueError('apply for random_contrast must be a bool.')
        random_contrast_apply = \
        preprocessing_config['data_augmentation']['random_contrast']['apply']
        if random_contrast_apply:
            options = preprocessing_config['data_augmentation'][
                'random_contrast']
            features['image/encoded'] = random_contrast(features[
                                                            'image/encoded'],
                                                        options)
        else:
            tf.logging.info('No random contrast augmentation applied.')

        if not isinstance(
                preprocessing_config['data_augmentation']['random_hue'][
                    'apply'], bool):
            raise ValueError('apply for random_hue must be a bool.')
        random_hue_apply = \
        preprocessing_config['data_augmentation']['random_hue']['apply']
        if random_hue_apply:
            options = preprocessing_config['data_augmentation']['random_hue']
            features['image/encoded'] = random_hue(features['image/encoded'],
                                                   options)
        else:
            tf.logging.info('No random_hue data augmentation applied.')

        if not isinstance(
                preprocessing_config['data_augmentation']['random_saturation'][
                    'apply'],
                bool):
            raise ValueError('apply for random_saturation must be a bool.')
        random_saturation_apply = \
        preprocessing_config['data_augmentation']['random_saturation'][
            'apply']
        if random_saturation_apply:
            options = preprocessing_config['data_augmentation'][
                'random_saturation']
            features['image/encoded'] = random_saturation(features[
                                                              'image/encoded'],
                                                          options)
        else:
            tf.logging.info('No random saturation augmentation applied.')

    label_dict = dict(
        object_labels=labels,
        object_bboxes=bboxes,
        original_shape=[features['image/height'], features['image/width'], 3],
        numbboxes=tf.shape(bboxes)[0]
    )
    label_dict = pad_input_data_to_static_shapes(label_dict, max_num_boxes,
                                                 numclasses)

    features['image/encoded'] = tf.squeeze(features['image/encoded'], axis=0)
    features['image/encoded'].set_shape((input_height, input_width, 3))

    return features['image/encoded'], label_dict


def flip_image_and_bounding_boxes(image, bboxes):
    with tf.variable_scope('flip_image_horizontal'):
        image = tf.image.flip_left_right(image)

    with tf.variable_scope('flip_boxes_horizontal'):
        ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4,
                                          axis=1)
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)

    return image, flipped_boxes


def random_brightness(image, options):
    if not isinstance(options['max_delta'], float) or options['max_delta'] < 0:
        raise ValueError('max_delta must be a non-negative float.')

    with tf.variable_scope('random_brightness'):
        image = tf.image.random_brightness(image, options['max_delta'])

    tf.logging.info('random_brightness augmentation applied with : '
                    'max_delta={}'.format(options['max_delta']))

    return image


def random_hue(image, options):
    if not isinstance(options['max_delta'], float):
        raise ValueError('max_delta for random_hue must be a float in the '
                         'range [0, 0.5).')

    max_delta = options['max_delta']
    if not 0. <= max_delta < 0.5:
        raise ValueError('max_delta for random_hue must be a float in the '
                         'range [0, 0.5).')

    with tf.variable_scope('random_hue'):
        image = tf.image.random_hue(image, max_delta)

    tf.logging.info(
        'random_hue augmentation applied with : max_delta={}.'.format(
            max_delta))

    return image


def random_contrast(image, options):
    lower = options['lower']
    upper = options['upper']

    for factor in [lower, upper]:
        if not isinstance(factor, float):
            raise ValueError('{} for random_contrast must be a float '
                             'value.'.format(factor))

    if lower < 0:
        raise ValueError('lower for random_contrast must be >=0.0')

    if upper <= lower:
        raise ValueError('upper<=lower is the condition that must be '
                         'satisfied for random_contrast.')

    with tf.variable_scope('random_contrast'):
        image = tf.image.random_contrast(image, lower=lower, upper=upper)

    tf.logging.info(
        'random_contrast augmentation applied with : lower={} and upper={}.'.format(
            lower, upper))
    return image


def random_saturation(image, options):
    lower = options['lower']
    upper = options['upper']

    for factor in [lower, upper]:
        if not isinstance(factor, float):
            raise ValueError('{} for random_saturation must be a float '
                             'value.'.format(factor))

    if lower < 0:
        raise ValueError('lower for random_saturation must be >=0.0')

    if upper <= lower:
        raise ValueError('upper<=lower is the condition that must be '
                         'satisfied for random_saturation.')

    with tf.variable_scope('random_saturation'):
        image = tf.image.random_saturation(image, lower=lower, upper=upper)

    tf.logging.info(
        'random_saturation augmentation applied with : lower={} and upper={}.'.format(
            lower, upper))
    return image


def pad_input_data_to_static_shapes(labels, max_gt_boxes, numclasses):
    padding_shapes = {
        'object_labels': [max_gt_boxes, numclasses],
        'object_bboxes': [max_gt_boxes, 4],

    }

    for key in padding_shapes.keys():
        labels[key] = pad_or_clip_nd(labels[key], padding_shapes[key])

    return labels


def pad_or_clip_nd(tensor, output_shape):
    tensor_shape = tf.shape(tensor)
    clip_size = [
        tf.where(tensor_shape[i] - shape > 0, shape, -1)
        if shape is not None else -1 for i, shape in enumerate(output_shape)
    ]
    clipped_tensor = tf.slice(
        tensor,
        begin=tf.zeros(len(clip_size), dtype=tf.int32),
        size=clip_size)

    # Pad tensor if the shape of clipped tensor is smaller than the expected
    # shape.
    clipped_tensor_shape = tf.shape(clipped_tensor)
    trailing_paddings = [
        shape - clipped_tensor_shape[i] if shape is not None else 0
        for i, shape in enumerate(output_shape)
    ]
    paddings = tf.stack(
        [
            tf.zeros(len(trailing_paddings), dtype=tf.int32),
            trailing_paddings
        ],
        axis=1)
    padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
    output_static_shape = [
        dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
    ]
    padded_tensor.set_shape(output_static_shape)

    return padded_tensor
