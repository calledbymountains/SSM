import tensorflow as tf
from data.datasets import create_dataset

MODE_MAPPING = dict(
    train='training',
    eval='validation',
    pred='testing'
)


def input_fn(config, mode):
    if mode not in ('train', 'eval', 'pred'):
        raise ValueError('mode must be one of "train", "eval" and "pred"')

    mode_map = MODE_MAPPING[mode]

    numclasses = config['data']['numclasses']
    if not isinstance(numclasses, int) or numclasses <= 0:
        raise ValueError('numclasses must be a positive integer.')

    tf.logging.info(
        'Creating the input function for the estimator for {}.'.format(
            mode_map))

    data_config = config['data'][mode_map]
    preprocessing_config = config['data']['preprocessing']

    dataset = create_dataset(data_config, numclasses, preprocessing_config,
                             mode)

    return dataset


