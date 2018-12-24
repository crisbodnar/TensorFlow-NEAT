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

import pickle

import neat
import numpy as np
import tensorflow as tf

from tf_neat.activations import tanh_activation
from tf_neat.recurrent_net import RecurrentNet

if not tf.executing_eagerly():
    tf.enable_eager_execution()


def assert_almost_equal(x, y, tol):
    assert abs(x - y) < tol, "{!r} !~= {!r}".format(x, y)


def test_unconnected():
    assert tf.executing_eagerly()

    net = RecurrentNet(
        n_inputs=1,
        n_hidden=0,
        n_outputs=1,
        input_to_hidden=([], []),
        hidden_to_hidden=([], []),
        output_to_hidden=([], []),
        input_to_output=([], []),
        hidden_to_output=([], []),
        output_to_output=([], []),
        hidden_responses=[],
        output_responses=[1.0],
        hidden_biases=[],
        output_biases=[0],
    )

    result = net.activate([[0.2]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.outputs[0, 0], 0.5, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])

    result = net.activate([[0.4]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.outputs[0, 0], 0.5, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])


def test_simple():
    assert tf.executing_eagerly()

    net = RecurrentNet(
        n_inputs=1,
        n_hidden=0,
        n_outputs=1,
        input_to_hidden=([], []),
        hidden_to_hidden=([], []),
        output_to_hidden=([], []),
        input_to_output=([(0, 0)], [1.0]),
        hidden_to_output=([], []),
        output_to_output=([], []),
        hidden_responses=[],
        output_responses=[1.0],
        hidden_biases=[],
        output_biases=[0],
    )

    result = net.activate([[0.2]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.outputs[0, 0], 0.731, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])

    result = net.activate([[0.4]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.outputs[0, 0], 0.881, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])


def test_hidden():
    assert tf.executing_eagerly()

    net = RecurrentNet(
        n_inputs=1,
        n_hidden=1,
        n_outputs=1,
        input_to_hidden=([(0, 0)], [1.0]),
        hidden_to_hidden=([], []),
        output_to_hidden=([], []),
        input_to_output=([], []),
        hidden_to_output=([(0, 0)], [1.0]),
        output_to_output=([], []),
        hidden_responses=[1.0],
        output_responses=[1.0],
        hidden_biases=[0],
        output_biases=[0],
        use_current_activs=True,
    )

    result = net.activate([[0.2]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.activs[0, 0], 0.731, 0.001)
    assert_almost_equal(net.outputs[0, 0], 0.975, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])

    result = net.activate([[0.4]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.activs[0, 0], 0.881, 0.001)
    assert_almost_equal(net.outputs[0, 0], 0.988, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])


def test_recurrent():
    assert tf.executing_eagerly()

    net = RecurrentNet(
        n_inputs=1,
        n_hidden=1,
        n_outputs=1,
        input_to_hidden=([(0, 0)], [1.0]),
        hidden_to_hidden=([(0, 0)], [2.0]),
        output_to_hidden=([], []),
        input_to_output=([], []),
        hidden_to_output=([(0, 0)], [1.0]),
        output_to_output=([], []),
        hidden_responses=[1.0],
        output_responses=[1.0],
        hidden_biases=[0],
        output_biases=[0],
        use_current_activs=True,
    )

    result = net.activate([[0.2]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.activs[0, 0], 0.731, 0.001)
    assert_almost_equal(net.outputs[0, 0], 0.975, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])

    result = net.activate([[-1.4]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.activs[0, 0], 0.577, 0.001)
    assert_almost_equal(net.outputs[0, 0], 0.947, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])


def test_dtype():
    assert tf.executing_eagerly()

    net = RecurrentNet(
        n_inputs=1,
        n_hidden=1,
        n_outputs=1,
        input_to_hidden=([(0, 0)], [1.0]),
        hidden_to_hidden=([(0, 0)], [2.0]),
        output_to_hidden=([], []),
        input_to_output=([], []),
        hidden_to_output=([(0, 0)], [1.0]),
        output_to_output=([], []),
        hidden_responses=[1.0],
        output_responses=[1.0],
        hidden_biases=[0],
        output_biases=[0],
        use_current_activs=True,
        dtype=tf.float32,
    )

    result = net.activate([[0.2]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.activs[0, 0], 0.731, 0.001)
    assert_almost_equal(net.outputs[0, 0], 0.975, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])

    result = net.activate([[-1.4]])
    assert result.shape == (1, 1)
    assert_almost_equal(net.activs[0, 0], 0.577, 0.001)
    assert_almost_equal(net.outputs[0, 0], 0.947, 0.001)
    assert tf.math.equal(result[0, 0], net.outputs[0, 0])


def test_match_neat():
    assert tf.executing_eagerly()

    with open("tests/test-genome.pkl", "rb") as f:
        genome = pickle.load(f)

    # use tanh since neat sets output nodes with no inputs to 0
    # (sigmoid would output 0.5 for us)
    def neat_tanh_activation(z):
        return float(tf.tanh(2.5 * tf.convert_to_tensor(z, dtype=tf.float64)))

    for node in genome.nodes.values():
        node.response = 0.5

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "tests/test-config.cfg",
    )

    for _ in range(500):
        genome.mutate(config.genome_config)
        # print(genome)

        neat_net = neat.nn.RecurrentNetwork.create(genome, config)
        for i, (node, _activation, aggregation, bias, response, links) in enumerate(
            neat_net.node_evals
        ):
            neat_net.node_evals[i] = (
                node,
                neat_tanh_activation,
                aggregation,
                bias,
                response,
                links,
            )

        tf_net = RecurrentNet.create(
            genome, config, activation=tanh_activation, prune_empty=True
        )

        for _ in range(5):
            inputs = np.random.randn(12)
            # print(inputs)
            neat_result = neat_net.activate(inputs)
            tf_result = tf_net.activate([inputs])[0].numpy()
            assert np.allclose(neat_result, tf_result, atol=1e-8)
