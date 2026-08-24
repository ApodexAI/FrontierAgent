ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

RUN python -m pip install --no-cache-dir "transformers[serving]>=5.0,<6" accelerate

ENTRYPOINT ["transformers", "serve"]
