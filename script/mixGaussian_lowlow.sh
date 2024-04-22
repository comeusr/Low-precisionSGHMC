#!/bin/bash -l


python ../gaussian/gaussian_mixture_model.py --dynamic 'hmc' --lr 0.01 --nsample 1000000 --U 7 --gamma 4 --precision 'low_low' --batch_size 500 --FL 4 --WL 8