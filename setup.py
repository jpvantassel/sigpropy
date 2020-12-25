
"""A setuptools based setup module."""

from setuptools import setup, find_packages

meta = {}
with open("sigpropy/meta.py") as f:
    exec(f.read(), meta)

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='sigpropy',
    version=meta['__version__'],
    description='A Python package for digital signal processing.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/signal-processing',
    author='Joseph P. Vantassel',
    author_email='jvantassel@utexas.edu',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='signal-processing signal',
    packages=find_packages(),
    python_requires='>=3.6, <3.9',
    install_requires=['scipy', 'numpy', 'obspy', 'numba'],
    extras_require={
        'dev': ['hypothesis', 'pandas', 'coverage'],
    },
    package_data={
    },
    data_files=[
    ],
    entry_points={
    },
    project_urls={
        'Bug Reports': 'https://github.com/jpvantassel/signal-processing/issues',
        'Source': 'https://github.com/jpvantassel/signal-processing',
        'Docs': 'https://sigpropy.readthedocs.io/en/latest/?badge=latest',
    },
)
