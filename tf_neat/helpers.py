import tensorflow as tf


def expand(tensor, multiples):
    m = []
    for i, dim in enumerate(tensor.shape):
        if dim == 1:
            m.append(multiples[i])
        else:
            m.append(1)
    return tf.tile(tensor, multiples=m)
