from setuptools import setup, find_packages
PACKAGES = find_packages()

from os import path
this_directory = path.abspath(path.dirname(__file__))

opts = dict(name='pte_design',
            maintainer='Andrew Fiore-Gartland',
            maintainer_email='agartlan@fredhutch.org',
            description='Generate peptides to be used for epitope mapping, maximizing coverage of a sequence alignment.',
            url='https://github.com/FredHutch/pte_design',
            license='MIT',
            author='Andrew Fiore-Gartland',
            author_email='agartlan@fredhutch.org',
            version='0.1',
            packages=PACKAGES
           )

install_reqs = [
      'numpy',
      'pandas',
      'pwseqdist',
      'matplotlib']

if __name__ == "__main__":
      setup(**opts, install_requires=install_reqs)
