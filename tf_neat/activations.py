# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
# This file was modified by github.com/crisbodnar to add TensorFlow support

import tensorflow as tf


def sigmoid_activation(x):
    return tf.sigmoid(5 * x)


def tanh_activation(x):
    return tf.tanh(2.5 * x)


def abs_activation(x):
    return tf.abs(x)


def gauss_activation(x):
    return tf.exp(-5.0 * x**2)


def identity_activation(x):
    return x


def sin_activation(x):
    return tf.sin(x)


def relu_activation(x):
    return tf.nn.relu(x)


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
}
