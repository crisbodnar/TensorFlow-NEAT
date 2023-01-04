from distutils.core import setup

setup(
    name="tf-neat",
    version="1.0",
    license="Apache License 2.0",
    description="An extension of NEAT-Python using TensorFlow",
    long_description=(
        "TensorFlow NEAT builds upon NEAT-Python by providing some functions which can"
        " turn a NEAT-Python genome into either a recurrent TensorFlow network or a"
        " TensorFlow CPPN for use in HyperNEAT or Adaptive HyperNEAT."
    ),
    author="Cristian Bodnar",
    maintainer_email="cb2015@cam.ac.uk",
    url="https://github.com/crisbodnar/TensorFlow-NEAT",
    packages=["tf_neat"],
    install_requires=[
        "neat-python>=0.92",
        "numpy>=1.14.3",
        "gym>=0.10.5",
        "click>=6.7",
        "tensorflow>=1.12.2",
        "keras-applications>=1.0.6",
        "keras-preprocessing>=1.0.5",
    ]
)
