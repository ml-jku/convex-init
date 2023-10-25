# Principled Weight Initialisation for Input-Convex Neural Networks 

Pieter-Jan Hoedt, Günter Klambauer

Input-Convex Neural Networks (ICNNs) [(Amos et al., 2017)](#icnn) can be hard to train.
This repository provides the implementation for an initialisation strategy that enables faster learning.
Moreover, it makes it possible to train ICNNs without skip-connections (cf. [Martens et al., 2021](#dks)).

On top of the initialisation itself, code for the different experiments and figures in the paper can be found here.

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
