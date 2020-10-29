FROM tensorflow/tensorflow:1.13.2-gpu-py3
# FROM tensorflow/tensorflow:1.13.2-gpu-py3-jupyter
MAINTAINER SongLei s974534426@gmail.com

# docker pull tensorflow/tensorflow:1.12.0-devel-gpu-py3

RUN apt-get update \
	&& pip config --global set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
	&& pip config --global set install.trusted-host mirrors.aliyun.com \
	&& pip install opencv-python==3.4.0.14 matplotlib pandas scipy \
	&& apt-get -y install apt-file \
	&& apt-file update \
	&& apt-get -y install libsm6 libxrender1 libxext6 \
	&& apt-get -y upgrade \
	&& apt-get -y install python-pydot python-pydot-ng graphviz \
	&& pip install pydot graphviz

CMD /bin/bash

docker build -t songlei/tf:1.13 .
sudo docker run -v $PWD:/data -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --gpus all -it songlei/tf:1.13 /bin/bash

sudo docker run --gpus all -it tensorflow/tensorflow:1.12.0-devel-gpu-py3 /bin/bash

sudo apt install llvm-9
export LLVM_CONFIG="/usr/bin/llvm-config-9"