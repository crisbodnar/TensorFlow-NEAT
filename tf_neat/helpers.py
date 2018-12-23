import tensorflow as tf


def shape(tensor):
    return tensor.get_shape().as_list()


def expand(tensor, multiples):
    m = []
    for i, dim in enumerate(shape(tensor)):
        if dim == 1:
            m.append(multiples[i])
        else:
            m.append(1)
    return tf.tile(tensor, multiples=m)
