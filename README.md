# Triton inference server CC

## Setup

> **IMPORTANT**: Check the supported triton version from the [Deep Learning Frameworks support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) and update the base image in Dockerfiles accordingly.
To get the version, use `nvcc --version`. For me it was CUDA:11.7 and I used 22.08 triton version.

1. Create the image

   ```bash
   docker build -t triton_cc:0.0.1 -f dockers/Dockerfile.cpu .
   ```

1. Run the notebook and save the `weights` folder to ensure the default pytorch model get's loaded.

1. Run the docker container with tritonserver in detach mode

   ```bash
   docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 --rm -it -v ${PWD}/models/:/project/models/ -v ${PWD}/weights/:/project/weights/ triton_cc:0.0.1 tritonserver --model-repository models/ --model-control-mode=poll
   ```

### Notes:

1. The current base image in dockerfile has all the backends, even including tensorflow which is not required. Consider [customizing it](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker). More info [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

1. The ports are for:

   1. http: 8000
   1. grpc: 8001
   1. metrics: 8002

1. We are mounting local models folder to container's models folder to ensure all the model files we create from within the notebook get's mapped automatically to the container and we can then ping tritionserver directly.

## Learnings for specific backend:

### Python backend.

1. Spend some time to debug the `model.py` code. You can add breakpoints and fix issues conviniently. Run the server and when you hit it using tritonclient, the breakpoint will work.
1. Always take care of the input dimension you are mentioning in the `config.pbtxt`.

### Onnx backend

1. Make sure the first axis of input and output is dynamic.
1. Always take care of the input dimension you are mentioning in the `config.pbtxt`.
1. Make sure to use the same input and output names while creating the onnx model and during client inference.
1. Take care of the dtypes you are using to compile to onnx and the onces specified in the config.pbtxt. For instance, incase of transformers tokenizer, it returns dtype int64 and if you use int32 (preferred) in config.pbtxt, it will fail.

### TensorRT Backend

#### Installation

Personal recommendation is to run this within a docker container.

1. Create a container:
   ```bash
   docker run --gpus all --rm -it -v ${PWD}/models/:/workspace/models/ -v ${PWD}/weights/:/workspace/weights/ -v ${PWD}/inference_notebook/:/workspace/inference_notebook/ --name triton_trtc nvcr.io/nvidia/tensorrt:22.08-py3
   ```

2. Run the TensorRT section of the notebook.

#### Useful info

1. TensorRT is not supported for each operation and can cause issues. In that case, try upgrading it's version but keep in mind the CUDA version and trition of your system. If possible update the CUDA version.
2. FP16 version takes time to compile so take a break.