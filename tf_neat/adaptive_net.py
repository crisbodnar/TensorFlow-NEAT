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
from .activations import tanh_activation
from .cppn import create_cppn, clamp_weights_, get_coord_inputs
from .helpers import expand


class AdaptiveNet:
    def __init__(self,
                 w_ih_node,
                 b_h_node,
                 w_hh_node,
                 b_o_node,
                 w_ho_node,
                 delta_w_node,
                 #  stateful_node,

                 input_coords,
                 hidden_coords,
                 output_coords,

                 weight_threshold=0.2,
                 activation=tanh_activation,

                 batch_size=1,
                 device='gpu'):

        with tf.device(device):
            self.w_ih_node = w_ih_node

            self.b_h_node = b_h_node
            self.w_hh_node = w_hh_node

            self.b_o_node = b_o_node
            self.w_ho_node = w_ho_node

            self.delta_w_node = delta_w_node
            # self.stateful_node = stateful_node

            self.n_inputs = len(input_coords)
            self.input_coords = tf.convert_to_tensor(input_coords, preferred_dtype=tf.float32)

            self.n_hidden = len(hidden_coords)
            self.hidden_coords = tf.convert_to_tensor(hidden_coords, preferred_dtype=tf.float32)

            self.n_outputs = len(output_coords)
            self.output_coords = tf.convert_to_tensor(output_coords, preferred_dtype=tf.float32)

            self.weight_threshold = weight_threshold

            self.activation = activation

            self.batch_size = batch_size
            self.device = device
            self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        with tf.device(self.device):
            (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

            n_in = len(in_coords)
            n_out = len(out_coords)

            zeros = tf.zeros((n_out, n_in), dtype=tf.float32)

            weights = w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in,
                             pre=zeros, post=zeros, w=zeros)
            return clamp_weights_(weights, self.weight_threshold)

    def reset(self):
        with tf.device(self.device):
            self.input_to_hidden = self.get_init_weights(self.input_coords, self.hidden_coords, self.w_ih_node)

            bias_coords = tf.zeros((1, 2), dtype=tf.float32)
            self.bias_hidden = self.get_init_weights(bias_coords, self.hidden_coords, self.b_h_node)
            self.bias_hidden = expand(tf.expand_dims(self.bias_hidden, 0), (self.batch_size, self.n_hidden, 1))

            self.hidden_to_hidden = self.get_init_weights(self.hidden_coords, self.hidden_coords, self.w_hh_node)
            self.hidden_to_hidden = expand(tf.expand_dims(self.hidden_to_hidden, 0),
                                           (self.batch_size, self.n_hidden, self.n_hidden))

            bias_coords = tf.zeros((1, 2), dtype=f.float32)
            self.bias_output = self.get_init_weights(bias_coords, self.output_coords, self.b_o_node)
            self.hidden_to_output = self.get_init_weights(self.hidden_coords, self.output_coords, self.w_ho_node)
            self.hidden = tf.zeros((self.batch_size, self.n_hidden, 1), dtype=tf.float32)

            self.batched_hidden_coords = get_coord_inputs(self.hidden_coords, self.hidden_coords,
                                                          batch_size=self.batch_size)

    def activate(self, inputs):
        '''
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        '''
        with tf.device(self.device):
            inputs = tf.expand_dims(tf.convert_to_tensor(inputs, dtype=tf.float32), 2)
            self.hidden = self.activation(self.input_to_hidden.matmul(inputs) +
                                          self.hidden_to_hidden.matmul(self.hidden) +
                                          self.bias_hidden)

            outputs = self.activation(self.hidden_to_output @ self.hidden + self.bias_output)

            hidden_outputs = expand(self.hidden, (self.batch_size, self.n_hidden, self.n_hidden))
            hidden_inputs = tf.transpose(hidden_outputs, perm=[0, 2, 1])

            (x_out, y_out), (x_in, y_in) = self.batched_hidden_coords

            self.hidden_to_hidden += self.delta_w_node(
                x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in,
                pre=hidden_inputs, post=hidden_outputs,
                w=self.hidden_to_hidden)

            return tf.squeeze(outputs, 2)

    @staticmethod
    def create(genome,
               config,

               input_coords,
               hidden_coords,
               output_coords,

               weight_threshold=0.2,
               activation=tanh_activation,
               batch_size=1,
               device='cuda:0'):

        nodes = create_cppn(
            genome, config,
            ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],
            ['w_ih', 'b_h', 'w_hh', 'b_o', 'w_ho', 'delta_w'])

        w_ih_node = nodes[0]
        b_h_node = nodes[1]
        w_hh_node = nodes[2]
        b_o_node = nodes[3]
        w_ho_node = nodes[4]
        delta_w_node = nodes[5]

        return AdaptiveNet(w_ih_node,
                           b_h_node,
                           w_hh_node,
                           b_o_node,
                           w_ho_node,
                           delta_w_node,

                           input_coords,
                           hidden_coords,
                           output_coords,

                           weight_threshold=weight_threshold,
                           activation=activation,
                           batch_size=batch_size,
                           device=device)
