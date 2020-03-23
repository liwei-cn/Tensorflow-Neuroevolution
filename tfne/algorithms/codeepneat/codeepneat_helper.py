import tensorflow as tf


def deserialize_merge_method(input):
    """"""
    deserialized_list = []
    for merge_method in input:
        if merge_method == 'concat':
            deserialized_list.append(tf.concat)
        elif merge_method == 'sum':
            deserialized_list.append(tf.math.reduce_sum)
        else:
            raise NotImplementedError("Supplied possible merge method '{}' not implemented".format(merge_method))
    return deserialized_list


def round_to_nearest_multiple(value, min, max, multiple):
    """"""
    lower_multiple = int(value / multiple) * multiple
    if value % multiple - (multiple / 2.0) < 0:
        if lower_multiple >= min and lower_multiple <= max:
            return lower_multiple
        if lower_multiple < min:
            return min
        if lower_multiple > max:
            return max
    else:
        higher_multiple = lower_multiple + multiple
        if higher_multiple >= min and higher_multiple <= max:
            return higher_multiple
        if higher_multiple < min:
            return min
        if higher_multiple > max:
            return max
