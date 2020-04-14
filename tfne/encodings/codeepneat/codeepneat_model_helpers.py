from typing import Callable

import tensorflow as tf


def deserialize_merge_method(merge_method_str) -> Callable:
    """"""
    if merge_method_str == 'concat':
        return tf.concat
    elif merge_method_str == 'sum':
        return tf.math.reduce_sum
    else:
        raise NotImplementedError("Config supplied possible merge method '{}' not implemented".format(merge_method_str))
