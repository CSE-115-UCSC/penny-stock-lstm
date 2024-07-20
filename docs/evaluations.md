1. Evaluate a new ticker that the model has never seen
   split into seen and unseen ticker

2. How to evaluate a ticker that a model has already seen?
   split each ticker on days.
   seen and unseen days.

# Tasks

1. Predict the daily sequence (almost done) (input 930-1030, output 1030-rest)
2. Should we buy a ticker (maybe a FFN network with fixed size input of first hour data and outputs a label: yes/no). For this the evaluation would be predict on the test set, check the test sequences, if the price actually went higher, ......
