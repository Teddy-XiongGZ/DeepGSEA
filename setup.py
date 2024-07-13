from setuptools import setup, find_packages
from src.__init__ import __version__, __author__, __email__
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    python_requires='>=3.9',
    name='DeepGSEA',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT License",

    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,

    url='https://github.com/Teddy-XiongGZ/DeepGSEA',

    packages=find_packages(include=['src.*']),
    package_dir={'':'src'},

    install_requires=required,

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ]
)
