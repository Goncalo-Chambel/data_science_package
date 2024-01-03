from setuptools import setup

setup(name='data_science',
      packages=['data_science'],
      version='0.0.1dev1',
      entry_points={
          'console_scripts': ['data_science-cli=data_science.cmd:main']
      }
    )