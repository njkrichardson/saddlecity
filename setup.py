import setuptools

setuptools.setup(name='saddlecity',
      version='0.1dev',
      description='Nonlinear dynamical systems theory for models of games, biological memory, and switching autoregressive processes.',
      author='Nick Richardson and Yoni Maltsman',
      author_email='nrichardson@hmc.edu, jmaltsman@hmc.edu',
      url='https://github.com/njkrichardson/saddlecity',
      long_description=open('README.md').read(), 
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
      python_requires=">=3.6"
     )
