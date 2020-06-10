# CPU - Ubuntu
FROM ubuntu:16.04
MAINTAINER Dibya Ghosh <@berkeley.edu>

# Copied from Vitchyr

# Remove apt-util warning for Ubuntu 16.05
# https://github.com/phusion/baseimage-docker/issues/319
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-2.5.0-Linux-x86_64.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Special Deps ==========
RUN apt-get update --fix-missing && apt-get -y install git make cmake unzip
RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
# MUJOCO requires graphics stuff (Why?)
RUN cp /etc/apt/sources.list /etc/apt/sources.list~
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN apt-get install -y vim ack-grep
RUN pip install --upgrade pip
# usual pip install pygame will fail
RUN apt-get build-dep -y python-pygame
RUN pip install Pillow

# ========== OpenAI Gym ==========
RUN apt-get -y install libgtk2.0-0
# RUN pip install gym
RUN alias ffmpeg="avconv"
# See https://github.com/openai/gym/issues/218
RUN apt-get -y install \
    xvfb \
    libav-tools \
    xorg-dev \
    libsdl2-dev \
    swig

# VizDoom
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    g++ \
    git \
    make \
    nasm \
    openssh-server \
    tar \
    timidity \
    vim \
    zip
    
# ========== mujoco-py ==========
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
    libosmesa6-dev

# ========== Add codebase stub ==========
RUN mkdir /root/code
ADD environment.yml /root/code/environment.yml
RUN cat /root/code/environment.yml
RUN conda env create -f /root/code/environment.yml

# This doesn't seem to work always. Not sure why.
# I'll just add railrl-rl env to the path manually and leave this here in case
# someone wants to fix this later
RUN echo "source activate gcsl" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

# gpu theanno
ENV THEANO_FLAGS mode=FAST_RUN,device=gpu,floatX=float32

ENV PATH /opt/conda/envs/gcsl/bin:$PATH

# More mujoco stuff
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
RUN /usr/bin/curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
RUN echo "Hello"
COPY .mujoco /root/.mujoco

# Trigger the cython code generation
RUN pip install -U 'lockfile'
RUN pip install -U 'mujoco-py<2.1,>=2.0'
RUN pip install -U 'pyparsing'
RUN pip install -U 'transforms3d'
RUN pip install -U 'box2d-py'
# RUN python -c "import mujoco_py"
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
RUN rm -rf /root/.mujoco/mjkey.txt

