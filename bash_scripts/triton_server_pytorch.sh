docker run --shm-size 2gb -p 8000:8000 -p 8001:8001 -p 8002:8002 --rm -it -v ${PWD}/models/:/project/models/ -v ${PWD}/weights/:/project/weights/ triton_cc_pt:0.0.1 tritonserver --model-repository models/ --log-verbose=1 --model-control-mode=poll
