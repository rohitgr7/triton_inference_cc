docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 --rm -it -v ${PWD}/models_tf/:/project/models_tf/ -v ${PWD}/weights_tf/:/project/weights_tf/ triton_cc_tf:0.0.1 tritonserver --model-repository models_tf/ --log-verbose=1 --model-control-mode=poll
