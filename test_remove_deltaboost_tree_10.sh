#!/usr/bin/env bash

n_rounds=$1  # 1, 10, 30, 100

for dataset in cadata codrna gisette covtype msd; do
  bash test_remove_deltaboost.sh 10 $dataset 1e-03 $n_rounds
  bash test_remove_deltaboost.sh 10 $dataset 1e-02 $n_rounds
  bash test_remove_gbdt.sh 10 $dataset
done
wait
