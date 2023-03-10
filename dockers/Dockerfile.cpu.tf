FROM nvcr.io/nvidia/tritonserver:22.08-py3

ARG PROJECT_PATH=/project

WORKDIR ${PROJECT_PATH}
SHELL ["/bin/bash", "-c"]

COPY requirements ${PROJECT_PATH}/requirements/

RUN pip install --upgrade pip && \
    pip install tensorflow-cpu==2.11.0 && \
    pip install -r ${PROJECT_PATH}/requirements/tf.txt
