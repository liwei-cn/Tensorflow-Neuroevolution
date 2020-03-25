import tensorflow as tf


def deserialize_merge_method(input_list):
    """"""
    deserialized_list = []
    for merge_method in input_list:
        if merge_method == 'concat':
            deserialized_list.append(tf.concat)
        elif merge_method == 'sum':
            deserialized_list.append(tf.math.reduce_sum)
        else:
            raise NotImplementedError("Supplied possible merge method '{}' not implemented".format(merge_method))
    return deserialized_list


def round_to_nearest_multiple(value, minimum, maximum, multiple):
    """"""
    lower_multiple = int(value / multiple) * multiple
    if value % multiple - (multiple / 2.0) < 0:
        if minimum <= lower_multiple <= maximum:
            return lower_multiple
        if lower_multiple < minimum:
            return minimum
        if lower_multiple > maximum:
            return maximum
    else:
        higher_multiple = lower_multiple + multiple
        if minimum <= higher_multiple <= maximum:
            return higher_multiple
        if higher_multiple < minimum:
            return minimum
        if higher_multiple > maximum:
            return maximum
