import tensorflow as tf
import rnn
RNN_NAME_TO_METHODS = dict(
    lstm=rnn.make_lstm,
    bidirectional_lstm=rnn.make_bidirectional_lstm,
    gru=rnn.make_gru
)