## Triton inference server CC

### Setup

1. Define the dockerfile
1. Create the image

```bash
docker build -t triton_cc:0.0.1 -f Dockerfile .
```

3. Run the docker container with tritonserver in detach mode

```bash
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 --rm -it -v ${PWD}/models/:/project/models/ triton_cc:0.0.1 tritonserver --model-repository models/
```

#### Notes:

1. The current base image in dockerfile has all the backends, even including tensorflow which is not required. Consider [customizing it](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker). More info [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

1. The ports are for:

   1. http: 8000
   1. grpc: 8001
   1. metrics: 8002

1. We are mounting local models folder to container's models folder to ensure all the model files we create from within the notebook get's mapped automatically to the container and we can then ping tritionserver directly.

### Learnings for specific backend:

#### Python backend.

1. Spend some time to debug the `model.py` code. You can add breakpoints and fix issues conviniently. Run the server and when you hit it using tritonclient, the breakpoint will work.
1. Always take care of the input dimension you are mentioning in the `config.pbtxt`.

#### Onnx backend

1. Make sure the first axis of input and output is dynamic.
1. Always take care of the input dimension you are mentioning in the `config.pbtxt`.
1. Make sure to use the same input and output names while creating the onnx model and during client inference.
1. Take care of the dtypes you are using to compile to onnx and the onces specified in the config.pbtxt. For instance, incase of transformers tokenizer, it returns dtype int64 and if you use int32 (preferred) in config.pbtxt, it will fail.
