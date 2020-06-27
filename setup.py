from setuptools import setup, find_packages

setup(
    name="CellCnn",
    version="0.3",
    author="Eirini Arvaniti",
    description="Convolutional neural network for analyzing single cell measurements",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "seaborn",
        "Keras>=2.3",
        "statsmodels",
        "FlowIO",
        "tensorflow>=2.2"
    ]
)
