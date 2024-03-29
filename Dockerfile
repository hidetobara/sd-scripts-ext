#FROM alpine:3.17 as xformers
#RUN apk add --no-cache aria2
#RUN aria2c -x 5 --dir / --out wheel.whl 'https://github.com/AbdBarho/stable-diffusion-webui-docker/releases/download/5.0.0/xformers-0.0.17.dev449-cp310-cp310-manylinux2014_x86_64.whl'

#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt -y update && apt -y install git vim less python3-pip wget curl libgl1-mesa-dev libglib2.0-0
#libopencv-dev

#RUN pip3 install torch==1.13.1+cu117 torchvision --extra-index-url https://download.pytorch.org/whl/cu117
#RUN pip3 install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
#ENV DOCKER_BUILDKIT=1
#RUN --mount=type=cache,target=/root/.cache/pip  \
#  --mount=type=bind,from=xformers,source=/wheel.whl,target=/xformers-0.0.15-cp310-cp310-linux_x86_64.whl \
#  pip3 install triton /xformers-0.0.15-cp310-cp310-linux_x86_64.whl
#RUN pip3 install git+https://github.com/facebookresearch/xformers.git@xformers

WORKDIR /app
COPY ./sd-scripts /app
RUN pip3 install -r requirements.txt

# sudo docker build -t tuning .
# sudo docker run --shm-size=256m --gpus all -it --rm -v ~/sd-scripts:/app -v ~/stable-diffusion-xl-docker/models:/models tuning /bin/bash
