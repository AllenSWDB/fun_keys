# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    ipywidgets \
    numpy==1.23.4 \
    scikit-learn==1.3.0 \
    allensdk==2.15.1 \
    dynamax[notebooks] \
    plotly \
    ipympl \
    pyarrow

ENV PYTHONPATH "${PYTHONPATH}:/root/capsule/code"