# Base image
FROM quay.io/fenicsproject/dev:latest

#RUN wget https://api.github.com/repos/pyamg/pyamg/tarball/v4.2.3 -O pyamg.tgz \
#    && tar xf pyamg.tgz \
#    && cd pyamg-pyamg-*/ \
#    && python3 -m pip install --user --upgrade pybind11 \
#    && python3 setup.py install --user

# cbc.block
COPY . .
RUN python3 setup.py install

# OR to run "in-place":
#
# docker run -it --rm -u $(id -u):$(id -g) -v /root:$PWD dolfinx/dolfinx python3 demo/mixedpoission.py
