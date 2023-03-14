# Base image
FROM ghcr.io/scientificcomputing/fenics:2023-03-01

WORKDIR /src/
RUN mkdir -p /src/cbc.block
COPY . /src/cbc.block/

ARG HAZMATH_VERSION=v1.0.1
ARG PYAMG_VERSION=v4.2.3

ENV SUITESPARSE_DIR=/usr/local/petsc/linux-gnu-real-32/
ENV PYTHONPATH=""
ENV LD_LIBRARY_PATH=/usr/local/petsc/linux-gnu-real-32/lib/:{LD_LIBRARY_PATH}
ENV HAZMATH_VERSION=$HAZMATH_VERSION
ENV PYAMG_VERSION=$PYAMG_VERSION

RUN git clone --branch=${PYAMG_VERSION} --single-branch --depth=1 https://github.com/pyamg/pyamg.git && \
    python3 -m pip install pyamg

# Haznics
RUN apt-get update && apt-get install -y swig && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/tmp/*

RUN git clone --branch=${HAZMATH_VERSION} --single-branch --depth=1 https://github.com/HAZmathTeam/hazmath && \
    cd hazmath && \ 
    make config shared=yes suitesparse=yes lapack=yes haznics=yes swig=yes . && \
    make install &&\
    cp -a swig_files haznics && \
    mv haznics/haznics.py haznics/__init__.py


# NOTE: This is incredibly hacky, and we should change hazmath to be a proper library
WORKDIR /src/hazmath
RUN echo "from distutils.core import setup\n\
setup(name='haznics', version='${HAZMATH_VERSION}', packages=['haznics'],\n\
package_data={'haznics': ['_haznics.so']})\n\
    " > setup.py

RUN python3 -m pip install . && \
    cp lib/libhazmath.so /usr/lib/

# cbc.block
WORKDIR /src
COPY . /src/cbc.block/
RUN python3 -m pip install -e cbc.block/[test,docs,haznics]
