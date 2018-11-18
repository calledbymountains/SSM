import tensorflow as tf
from data.datasets import create_dataset
from utils.writeconfig import DEFAULT_CONFIG

tf.logging.set_verbosity(tf.logging.INFO)
db = create_dataset(DEFAULT_CONFIG['data']['training'], 80, DEFAULT_CONFIG[
    'data']['preprocessing'], 'train')

dbiter = db.make_initializable_iterator()

data = dbiter.get_next()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(dbiter.initializer)
    try:
        while True:
            data_out = sess.run(data)
            print(data['features'].shape)
            for key in data_out['labels']:
                print('{} - {}'.format(key, data_out['labels'][key].shape))
                if key == 'object_bboxes':
                    print(data_out['labels'][key][0, 0:data_out['labels'][
                        'numbboxes'][0]
                          , :])
    except tf.errors.OutOfRangeError:
        print('Finished.')
