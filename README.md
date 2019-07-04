## GPU Accelerated PCSR implementation with Python
## *Power-Constrained Image Contrast Enhancement Through Sparse Representation by Joint Mixed-Norm Regularization*
Jia-Li Yin, Bo-Hao Chen, En-Hung Lai, and Ling-Feng Shi

![](/demo.png)

## Prerequisites:
> * Linux
> * Anaconda
> * CUDA 9.2
> * cuDNN 7.2.1
> * Numbapro 0.23.1
> * Python 2.7
> * Numpy 1.10.4
> * pip 19.1.1
> * OpenCV 3.4.1

## It was tested and runs under the following OSs:
* Ubuntu 18.04
* Ubuntu 16.04
Might work under others, but didn't get to test any other OSs just yet.

## Getting Started:
### Installation
- create a test environment
```bash
pip conda create -n PCSR numbapro
source activate PCSR
```

### Testing 
- To test the model:
```bash
python main.py
``` 
The test results will be saved in: `./results/.`

## Citation:
    @ARTICLE{yin2019PCCE, 
    author={J. {Yin} and B. {Chen} and E. {Lai} and L. {Shi}}, 
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={Power-constrained Image Contrast Enhancement through Sparse Representation by Joint Mixed-norm Regularization}, 
    year={2019}, 
    volume={}, 
    number={}, 
    pages={1-1}, 
    keywords={Organic light emitting diodes;Image reconstruction;Power demand;Image quality;TV;Batteries;Computational modeling;contrast enhancement;sparse representation;power consumption}, 
    doi={10.1109/TCSVT.2019.2925208}, 
    ISSN={1051-8215}, 
    month={},}
