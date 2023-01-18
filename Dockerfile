FROM nvcr.io/nvidia/tritonserver:22.12-py3
# FROM nvcr.io/nvidia/tritonserver:21.08-pyt-python-py3

ARG PROJECT_PATH

ARG PROJECT_PATH=/project

WORKDIR ${PROJECT_PATH}
SHELL ["/bin/bash", "-c"]

COPY requirements.txt ${PROJECT_PATH}/requirements.txt

RUN pip install -r ${PROJECT_PATH}/requirements.txt
