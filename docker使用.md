## 1. Docker安装

参考官网：https://docs.docker.com/engine/install/ubuntu/

## 2. 基本命令

```docker ps -a```列出当前运行的容器。

```docker rm 编号```删除容器。

```docker rmi 编号```删除镜像。

## 2. 修改镜像存储位置

通过```sudo docker info | grep "Docker Root"```查看当前镜像路径。


