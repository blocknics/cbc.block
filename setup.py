import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
      name='cbc.block',
      version='1.0.1',
      scripts=['cbc.block'] ,
      author="Joachim Berdal Haga",
      author_email="jobh@simula.no",
      description="Block utilities for FENiCS/dolfin",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://bitbucket.org/fenics-apps/cbc.block/",
      packages=setuptools.find_packages(),
      classifiers=[
         "Programming Language :: Python :: 3",
         "License :: GNU GPLv3 License",
         "Operating System :: OS Independent",
      ],
 )

