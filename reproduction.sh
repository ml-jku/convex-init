#!/usr/bin/env bash

# This script should provide (almost) all of the commands necessary to reproduce the results in the paper.
# The main goal of this script is to document how the different commands can be invoked for reproduction.
# Note that these commands were run in parallel on different servers.
# Please, DO NOT just run this script, since it will take much longer than necessary.

# training experiments
alias train="python -m init_learnability.train --config init_learnability/default.yaml"
alias visualise="python -m init_learnability.visualise results/init_learnability"

for DATA in MNIST CIFAR10 CIFAR100; do
  for H in 3 5 7; do
    # data for FIGURE 5 (and thus 2)
    train data.name=$DATA model.num_hidden=$H;
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=false;
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=false model.skip = true;
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=true;

    # additional data for FIGURE 7
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=true model.rand_bias=true;
    # additional data for FIGURE 8 (without bug)
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=true model.bias_init_only=true;
    # additional data for FIGURE 9
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=true model.corr=0.1;
    train data.name=$DATA model.num_hidden=$H model.positivity=icnn model.better_init=true model.corr=0.9;
  done;
done;

visualise --rand_bias false --bias_init_only false --corr 0.5 --depth 5;             # FIGURE 2
visualise --rand_bias false --bias_init_only false --corr 0.5;                       # FIGURE 5
visualise --better_init true --skip false --bias_init_only false --corr 0.5;         # FIGURE 7
visualise --better_init true --skip false --rand_bias false --corr 0.5;              # FIGURE 8
visualise --better_init true --skip false --rand_bias false --bias_init_only false;  # FIGURE 9

for DATA in MNIST CIFAR10 CIFAR100; do
  # additional data for FIGURE 10
  train data.name=$DATA model={num_hidden: 5, positivity: exp, better_init: false};
  train data.name=$DATA model={num_hidden: 5, positivity: exp, better_init: false, skip: true};
  train data.name=$DATA model={num_hidden: 5, positivity: exp, better_init: true};
done;

visualise --convexity: exp;  # FIGURE 10


# generalisation experiments
python -m mlp_search.generate_configs

alias search="python -m mlp_search.search --config/system/local.yaml"
alias train="python -m mlp_search.train --data-root ~/.pytorch"  # assumes data is stored in ~/.pytorch
alias visualise="python -m mlp_search.visualise"

for CONFIGS in options/*/*/; do
  search $CONFIGS;  # starts multiple runs in parallel
done;
for BEST in results/best_*.yaml; do
  train $BEST --config $BEST;  # starts multiple runs in parallel
done;

visualise nn icnn skip ours;  # FIGURE 3
visualise nn exp1 exp2;  # FIGURE 11


# TOX21 experiments
alias train="python -m tox21.train --config tox21/default.yaml"
alias evaluate="ptyhon -m tox21.evaluate "

train model.name=fc
train model.name=icnn
train model.name=icnn model.bad_init=true
train model.name=icnn model.skip=true

# collect numbers for TABLE 1
for TIMESTAMP in results/tox21/*.0; do
  evaluate "$( basename ${TIMESTAMP%.0} )" >> table.txt;
done;


# visualisation of distributions
python visualise_sigprop.py  # histograms and correlation matrices for FIGURE 1 and FIGURE 4
# NOTE: the finished SVG files can be found in the project directory (propagation_diff*.svg)
