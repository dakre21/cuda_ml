from setuptools import setup

setup(name='cuda_ml',
      version='0.0.0',
      packages=['cuda_ml'],
      install_requires=[
          'numpy',
          'click',
          'scipy',
          'pycuda'
      ],
      entry_points={
          'console_scripts': [
              'cuda_ml = cuda_ml.__main__:main'
          ]
      },
)
