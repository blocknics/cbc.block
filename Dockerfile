# Base image
FROM ghcr.io/scientificcomputing/fenics:2023-03-01

WORKDIR /tmp/
RUN mkdir -p /tmp/cbc.block
COPY . /tmp/cbc.block/

ARG HAZMATH_VERSION=v1.0.1
ARG PYAMG_VERSION=v4.2.3

ENV SUITESPARSE_DIR=/usr/local/petsc/linux-gnu-real-32/
ENV PYTHONPATH=""
ENV HAZMATH_VERSION=$HAZMATH_VERSION
ENV PYAMG_VERSION=$PYAMG_VERSION

RUN git clone --branch=${PYAMG_VERSION} --single-branch --depth=1 https://github.com/pyamg/pyamg.git && \
    python3 -m pip install pyamg

# Haznics
RUN git clone --branch=${HAZMATH_VERSION} --single-branch --depth=1 https://github.com/HAZmathTeam/hazmath && \
    cd hazmath && \ 
    make config shared=yes suitesparse=yes lapack=yes haznics=yes swig=yes . && \
    make install

ENV PYTHONPATH /tmp/hazmath/swig_files:${PYTHONPATH}

# cbc.block
RUN python3 -m pip install cbc.block/[all]
