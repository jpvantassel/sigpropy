"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name='sigpropy',
    version='0.1.1',
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
    python_requires = '>=3.6, <3.9',
    install_requires=['scipy>=1.2.1', 'numpy>=1.16.2', 'obspy>=1.1.1', 'json'],
    extras_require={
        'dev': ['unittest', 'hypothesis'],
        'test': ['unittest', 'hypothesis'],
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
