from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'm6ATM - m6A Transcrtipome-wide Mapper'
LONG_DESCRIPTION = 'm6ATM is a deep learning based m6A detection model using Nanopore Direct RNA sequencing data.'

# settings
setup(name = 'm6atm', 
      version = VERSION,
      author = 'Boyi Yu',
      author_email = 'bo-yi.yu@genome.rcast.u-tokyo.ac.jp',
      description = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      packages = ['m6atm', 'm6atm.preprocess', 'm6atm.train', 'm6atm.model'],
      include_package_data = True,
      install_requires = ['biopython>=1.81', 
                          'dask[complete]>=2021.9.1',
                          'h5py>=2.10.0',
                          'npy-append-array>=0.9.13',
                          'numba>=0.56.4',
                          'numpy>=1.18.4',
                          'ont-fast5-api>=4.1.1',
                          'pandas>=1.3.5',
                          'pysam>=0.21.0',
                          'ruptures>=1.1.7',
                          'scikit-learn>=1.0.2',
                          'statsmodels>=0.13.5',
                          'tsaug>=0.2.1',
                          'matplotlib>=3.5.3',
                          'seaborn>=0.12.2',
                          'tqdm>=4.65.0'],
      
     entry_points = {'console_scripts':['m6atm = m6atm.main:main']})