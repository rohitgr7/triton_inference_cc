# Triton inference server CC

## Intro

This repo contains all the code I worked on while learning [Triton Inference Server](https://github.com/triton-inference-server/server). It has both PyTorch and Tensorflow pipeline but my main focus is to explore PyTorch models so all the advanced features will be based on PyTorch.

For starters, you can set up your environment as explained below and then walk through the starter notebooks:

```console
inference_notebooks/inference_pytorch.ipynb
inference_notebooks/inference_tensorflow.ipynb
```

Later you can explore advanced use cases in `inference_notebooks/advance.ipynb`.

I have also prepared some notes here in README, you can explore them too.

## Setup

> **IMPORTANT**: Check the supported triton version from the [Deep Learning Frameworks support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) and update the base image in Dockerfiles accordingly.
> To get the version, use `nvcc --version`. For me it was CUDA:11.7 and I used 22.08 triton version.

1. Create the image

   ```bash
   docker build -t triton_cc_pt:0.0.1 -f dockers/Dockerfile.cpu.pt .
   ```

   For tensorflow:

   ```bash
   docker build -t triton_cc_tf:0.0.1 -f dockers/Dockerfile.cpu.tf .
   ```

1. Run the notebook and save the `weights` folder to ensure the default PyTorch model gets loaded.

1. Run the docker container with tritonserver in detach mode

   ```bash
   bash bash_scripts/triton_server_pytorch.sh
   ```

   For tensorflow:

   ```bash
   bash bash_scripts/triton_server_tensorflow.sh
   ```

### Notes:

1. The current base image in dockerfile has all the backends, which may not be required. Consider [customizing it](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker). More info [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

1. The ports are for:

   1. http: 8000
   1. grpc: 8001
   1. metrics: 8002

1. We are mounting the local `models` folder to the container's `models` folder to ensure all the model files we create from within the notebook get mapped automatically to the container and we can then ping tritionserver directly.

## Learnings for specific backend:

### Python backend.

1. Spend some time debugging the `model.py` code. You can add breakpoints and fix issues conveniently. Run the server and when you hit it using tritonclient, the breakpoint will work.
1. Always take care of the input dimension you are mentioning in the `config.pbtxt`.

### Onnx backend

1. Make sure the first axis of input and output is dynamic for batch_size.
1. Always take care of the input dimension you are mentioning in the `config.pbtxt`.
1. Make sure to use the same input and output names while creating the Onnx model and during client inference.
1. Take care of the dtypes you are using to compile to Onnx and the ones specified in the `config.pbtxt`. For instance, in the case of transformers tokenizer, it returns dtype int64 and if you use int32 (preferred) in `config.pbtxt`, it will fail.

### TensorRT Backend

#### Installation

Personal recommendation is to run this within a docker container.

1. Create a container:

   ```bash
   docker run --gpus all --rm -it -v ${PWD}/models/:/workspace/models/ -v ${PWD}/weights/:/workspace/weights/ -v ${PWD}/inference_notebook/:/workspace/inference_notebook/ --name triton_trtc nvcr.io/nvidia/tensorrt:22.08-py3
   ```

1. Run the TensorRT section of the notebook.

#### Useful info

1. TensorRT is not supported for each operation and can cause issues. In that case, try upgrading its version but keep in mind the CUDA version and trition of your system. If possible update the CUDA version.
1. FP16 version takes time to compile so take a break.

## Features

### Dynamic Batching

1. While using the HTTP client, use asycn_infer and make sure to set concurrency while initializing the client.
1. While using the GRPC client, use async_infer with a callback. And don't use context manager with the client. Not sure what's the reason, but will explore and update here.

## TODO:

1. Performance analyser
1. Model analyzer
1. Metrics
1. Stable diffusion pipelines
1. Efficient deployment on the cloud (for eg. runpod.io)

## Resources

1. https://youtu.be/cKf-KxJVlzE
