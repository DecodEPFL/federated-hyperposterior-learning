import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PAC-PFL",
    version="0.0.1",
    description="Personalized Federated Learning of Probabilistic Models: A PAC-Bayesian Approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'server': 'server'},
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'gpytorch',
        'ipython',
        'matplotlib',
        'numpy',
        'pandas',
        'pyro-ppl',
        'torch',
        'seaborn',
        'scipy',
        'statsmodels'
    ],
)
