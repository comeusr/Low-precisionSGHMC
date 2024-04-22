#!/bin/bash -l


python ../gaussian/gaussian_mixture_model.py --dynamic 'hmc' --lr 0.01 --nsample 1000000 --U 7 --gamma 4 --precision 'low_full' --batch_size 1000 --FL 4 --WL 8