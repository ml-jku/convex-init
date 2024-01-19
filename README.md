# Principled Weight Initialisation for Input-Convex Neural Networks 

Pieter-Jan Hoedt, Günter Klambauer

Input-Convex Neural Networks (ICNNs) [(Amos et al., 2017)](#icnn) can be hard to train.
This repository provides the implementation for an initialisation strategy that enables faster learning.
Moreover, it makes it possible to train ICNNs without skip-connections (cf. [Martens et al., 2021](#dks)).

On top of the initialisation itself, code for the different experiments and figures in the paper can be found here.

### Structure

The top-level directory (in which this file resides) contains different kinds of files.
Our initialisation (`ConvexInitialiser`) is implemented as a standalone class in `convex_init.py`.
Implementations for a set of building blocks for ICNNs can be found in `convex_modules.py`.
The code for training a model and logging results has been collected in `trainer.py`.
Apart from some exceptions, most of the other files are scripts with initial attempts to get things to work.
These scripts should be directly runnable using a command like `python train_cifar.py`.

The code for the experiments in the paper is located in different directories / modules.
The `init_learnability` module focuses on training simple networks.
Experiments on the generalisation properties of the networks is provided in the `mlp_search` module.
Finally, the `tox21` module contains the code for experiments on Tox21.
The code for each of these modules is best called using `python -m init_learnability.train`.
For a full overview of the reproduction commands with arguments, we refer to the BASH script `reproduction.sh`.

Code is supposed to be run in an environment with the package versions in `requirements.txt` using python 3.9 or 3.10.

## Paper

[NeurIPS](https://neurips.cc/virtual/2023/poster/70408)

### Abstract

Input-Convex Neural Networks (ICNNs) are networks that guarantee convexity in their input-output mapping. 
These networks have been successfully applied for energy-based modelling, optimal transport problems and learning invariances.
The convexity of ICNNs is achieved by using non-decreasing convex activation functions and non-negative weights. 
Because of these peculiarities, previous initialisation strategies, which implicitly assume centred weights, are not effective for ICNNs. 
By studying signal propagation through layers with non-negative weights, we are able to derive a principled weight initialisation for ICNNs. 
Concretely, we generalise signal propagation theory by removing the assumption that weights are sampled from a centred distribution. 
In a set of experiments, we demonstrate that our principled initialisation effectively accelerates learning in ICNNs and leads to better generalisation. 
Moreover, we find that, in contrast to common belief, ICNNs can be trained without skip-connections when initialised correctly. 
Finally, we apply ICNNs to a real-world drug discovery task and show that they allow for more effective molecular latent space exploration.

### Citation

To cite this work, you can use the following bibtex entry:
 ```bib
@inproceedings{hoedt2023principled,
  title     = {Principled Weight Initialisation for Input-Convex Neural Networks},
  author    = {Hoedt, Pieter-Jan and Klambauer, G{\"u}nter Klambauer},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
  year      = {2023},
  url       = {https://openreview.net/forum?id=pWZ97hUQtQ}
}
```

## References

 - <span id="icnn">Amos, B., Xu, L. &amp; Kolter, J.Z.. (2017).</span> [Input Convex Neural Networks](https://proceedings.mlr.press/v70/amos17b.html). In <i>Proceedings of Machine Learning Research</i> 70:146-155.
 - <span id="dks">Martens, J., Ballard, A., Desjardins, G., Swirszcz, G., Dalibard, V., Sohl-Dickstein, J., &amp; Schoenholz, S. S. (2021)</span> [Rapid training of deep neural networks without skip connections or normalization layers using Deep Kernel Shaping](https://arxiv.org/abs/2110.01765). arXiv:2110.01765
