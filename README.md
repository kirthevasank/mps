# MPS
### Myopic Posterior Sampling for Adaptive Goal Oriented Design of Experiments



MPS is a general and flexible framework adaptive goal oriented design of
experiments.

In adaptive  design of experiments (DoE),
one wishes to design a sequence of experiments and
collect data so as to achieve a desired goal.
While there are many algorithms for specialised settings for adaptive DoE (such as
optimisation, active learning, level set estimation etc.), MPS aims to provide
a general framework that encompasses a broad variety of the problems.

To do so, one must specifiy their goal via a reward function.
For more details, see our
[paper](http://www.cs.cmu.edu/~kkandasa/pubs/kandasamyICML19mps.pdf).


### Installation \& Getting Started

This library can be installed via the following commands.
```bash
$ git clone https://github.com/kirthevasank/mps
$ cd mps
$ python setup.py install
```
We have provided a few examples in the
[examples](examples) directory.
Simply `cd examples` and execute those examples via
`python al_linear_rbf.py` etc.


### Acknowledgements
Research and development of the methods in this package were funded by
the Toyota Research Institute, Accelerated Materials Design & Discovery (AMDD) program.


### Citation
If you use any part of this code in your work, please cite our
[ICML 2019 paper](http://www.cs.cmu.edu/~kkandasa/pubs/kandasamyICML19mps.pdf).

```
@inproceedings{kandasamy2019myopic,
  title={Myopic Posterior Sampling for Adaptive Goal Oriented Design of Experiments},
  author={Kandasamy, Kirthevasan and Neiswanger, Willie and Zhang, Reed and Krishnamurthy,
Akshay and Schneider, Jeff and Poczos, Barnabas},
  booktitle={International Conference on Machine Learning},
  pages={3222--3232},
  year={2019}
}
```

### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](LICENSE.txt).

For questions, please email kandasamy@cs.cmu.edu.

"Copyright 2018-2019 Kirthevasan Kandasamy"
