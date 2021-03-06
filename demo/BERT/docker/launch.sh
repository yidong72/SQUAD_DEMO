#!/bin/bash
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

docker run -it --rm \
    --name sample-bert \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -u $(id -u):$(id -g) \
    -v ${HOME}:/host/ \
    -v $1:/data/ \
    -v $(pwd)/../../:/workspace/TensorRT \
    -p 8888:8888 \
    sample-bert bash


