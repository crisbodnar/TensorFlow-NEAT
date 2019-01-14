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

from .activations import identity_activation, tanh_activation
from .cppn import clamp_weights_, create_cppn, get_coord_inputs
from .helpers import expand


class AdaptiveLinearNet:
    def __init__(
        self,
        delta_w_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="/cpu:0",
    ):

        with tf.device(device):
            self.delta_w_node = delta_w_node

            self.n_inputs = len(input_coords)
            self.input_coords = tf.convert_to_tensor(input_coords, dtype=tf.float32)

            self.n_outputs = len(output_coords)
            self.output_coords = tf.convert_to_tensor(output_coords, dtype=tf.float32)

            self.weight_threshold = weight_threshold
            self.weight_max = weight_max

            self.activation = activation
            self.cppn_activation = cppn_activation

            self.batch_size = batch_size
            self.device = device
            self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        with tf.device(self.device):

            (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

            n_in = in_coords.shape[0]
            n_out = out_coords.shape[0]

            zeros = tf.zeros((n_out, n_in), dtype=tf.float32)

            weights = self.cppn_activation(
                w_node(
                    x_out=x_out,
                    y_out=y_out,
                    x_in=x_in,
                    y_in=y_in,
                    pre=zeros,
                    post=zeros,
                    w=zeros,
                )
            )
            return clamp_weights_(weights, self.weight_threshold, self.weight_max)

    def reset(self):
        with tf.device(self.device):
            self.input_to_output = self.get_init_weights(self.input_coords, self.output_coords, self.delta_w_node)
            self.input_to_output = tf.expand_dims(self.input_to_output, 0)
            self.input_to_output = expand(self.input_to_output,
                                          multiples=(self.batch_size, self.n_outputs, self.n_inputs))

            self.w_expressed = tf.not_equal(self.input_to_output, tf.constant(0.0))

            self.batched_coords = get_coord_inputs(self.input_coords, self.output_coords, batch_size=self.batch_size)

    def activate(self, inputs):
        """
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        """
        with tf.device(self.device):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            inputs = tf.expand_dims(inputs, 2)

            outputs = self.activation(self.input_to_output @ inputs)

            input_activs = tf.transpose(inputs, perm=[0, 2, 1])
            input_activs = expand(input_activs, multiples=(self.batch_size, self.n_outputs, self.n_inputs))
            output_activs = expand(outputs, multiples=(self.batch_size, self.n_outputs, self.n_inputs))

            (x_out, y_out), (x_in, y_in) = self.batched_coords

            delta_w = self.cppn_activation(
                self.delta_w_node(
                    x_out=x_out,
                    y_out=y_out,
                    x_in=x_in,
                    y_in=y_in,
                    pre=input_activs,
                    post=output_activs,
                    w=self.input_to_output,
                )
            )

            self.delta_w = delta_w
            self.input_to_output = self.input_to_output.numpy()
            self.input_to_output[self.w_expressed] += self.delta_w.numpy()[self.w_expressed]
            self.input_to_output = clamp_weights_(self.input_to_output, weight_threshold=0.0, weight_max=self.weight_max)

            return tf.squeeze(outputs, 2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):

        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
            ["delta_w"],
            output_activation=output_activation,
        )

        delta_w_node = nodes[0]

        return AdaptiveLinearNet(
            delta_w_node,
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            batch_size=batch_size,
            device=device,
        )
