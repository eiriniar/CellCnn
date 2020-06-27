from setuptools import setup, find_packages

setup(
    name="CellCnn",
    version="0.3",
    author="Eirini Arvaniti",
    description="Convolutional neural network for analyzing single cell measurements",
    packages=find_packages(),
    install_requires=[
        "Keras>=2.3",
        "tensorflow>=2.2"
    ]
)
