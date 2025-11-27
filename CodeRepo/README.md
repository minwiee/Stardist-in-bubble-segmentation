## Installation

The code is compatible with python version 3.7. Please install the required packages using conda with the following procedure. At first, the packages listed in the requirements.txt file can be installed using `pip`

> pip install -r requirements.txt

Afterwards install mxnet using either the GPU enabled version (**recommended**)

> pip install mxnet-cu101==1.5.0

if you have a CUDA compatible GPU or the corresponding CPU version

> pip install mxnet==1.5.0

if you can not use GPU. For Linux also newer mxnet version are possible. Then the correct numpy version needs to be installed with

> pip install numpy==1.20.0

Ignore all incoming warnings.
