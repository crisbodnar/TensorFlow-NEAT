#!/usr/bin/env python

from distutils.core import setup

setup(name = "tensorflow-neat",
description = "Integrates NeuroEvolution of Augmenting Topologies with TensorFlow",
packages = ["tf_neat"],
package_dir = {"tf_neat": "tf_neat"}
)
