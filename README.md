# DeepSSM_SysID

Official repository for the PyTorch implementation of the paper: \
**Deep State Space Models for Nonlinear System Identification**, submitted to [CDC 2020](https://cdc2020.ieeecss.org/). \
Paper available on [[arXiv]](https://arxiv.org/pdf/2003.14162.pdf) with implemented [[code]](https://github.com/dgedon/DeepSSM_SysID/tree/master/). \
Authors: [Daniel Gedon](https://katalog.uu.se/profile/?id=N19-1795), [Niklas Wahlström](https://katalog.uu.se/profile/?id=N16-250), [Thomas B. Schön](http://user.it.uu.se/~thosc112/), [Lennart Ljung](http://users.isy.liu.se/rt/ljung/).

In this work we use six new deep State-Space Models (SSMs) developed from various authors in previous work and apply them for the field of nonlinear system identification. The available code provides a reimplementation of the six different models in PyTorch as a unified framework for these models. A toy problem and two established nonlinear benchmarks are used. The chosen methods benefit besides the identification of the system dynamics also from uncertainty quantification.     


If you find this work useful, please consider citing:
```
@article{gedonDeepStateSpace2020,
  title = {Deep {{State Space Models}} for {{Nonlinear System Identification}}},
  author = {Gedon, Daniel and Wahlstr{\"o}m, Niklas and Sch{\"o}n, Thomas B. and Ljung, Lennart},
  year = {2020},
  month = {March},
  archivePrefix = {arXiv},
  eprint = {2003.14162},
  eprinttype = {arxiv},
  journal = {arXiv:2003.14162 [cs, eess, stat]},
} 
```

## Repository overview

The different models are available in [/models](https://github.com/dgedon/DeepSSM_SysID/tree/master/models). The models are:
- VAE-RNN
- VRNN-Gauss-I
- VRNN-Gauss
- VRNN-GMM-I
- VRNN-GMM
- STORN 

The files of experiments to be able to generate the figures in the paper are available in 
[/final_toy_lgssm](https://github.com/dgedon/DeepSSM_SysID/tree/master/final_toy_lgssm),
[/final_narendra_li](https://github.com/dgedon/DeepSSM_SysID/tree/master/final_narendra_li) and
[/final_wiener_hammerstein](https://github.com/dgedon/DeepSSM_SysID/tree/master/final_wiener_hammerstein). 

To run a single model you can use the file in the folder [/experiment](https://github.com/dgedon/DeepSSM_SysID/tree/master/experiments)
called [main_single.py](https://github.com/dgedon/DeepSSM_SysID/blob/master/experiments/main_single.py). 
Within the option list you can choose a specific model and a specific dataset.

The used data files are stored in [/data](https://github.com/dgedon/DeepSSM_SysID/tree/master/data). 
For the Wiener Hammerstein system we refer to the original website (see readme in the folder) since the data files are rather large.
In order to extend for more datasets the dataset has to be provided in a specific format and added in the [/data/loader.py](https://github.com/dgedon/DeepSSM_SysID/blob/master/data/loader.py).
A training, validation and test dataset has to be provided as numpy arrays of shape (sequence length, signal dimension). 
The sequence length is defined in the file [/options/dataset_options.py](https://github.com/dgedon/DeepSSM_SysID/blob/master/options/dataset_options.py).
