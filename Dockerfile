# Base image
FROM quay.io/fenicsproject/dev

ENV PYTHONPATH=""

USER fenics

RUN git clone https://github.com/pyamg/pyamg.git && \
    cd pyamg && \
    python3 setup.py install --user && \
    cd ..

# cbc.block
RUN git clone https://mirok-w-simula@bitbucket.org/mirok-w-simula/cbc.block.git && \
    cd cbc.block && \
    python3 setup.py install --user && \
    cd ..

USER root
