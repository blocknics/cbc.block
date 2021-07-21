#!/usr/bin/env python

from distutils.core import setup
setup(name = "cbc.block",
      version = "0.0.1",
      description = "Block utilities",
      author = "Joachim Berdal Haga",
      author_email = "jobh@simula.no",
      url = "http://www.launchpad.net/cbc.block/",
      packages = ["block", "block.algebraic", "block.algebraic.trilinos", "block.iterative", "block.algebraic.petsc", "block.algebraic.hazmath"]
)

