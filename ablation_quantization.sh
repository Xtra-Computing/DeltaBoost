#!/usr/bin/env bash


dataset=$1  # cadata, codrna, gisette, covtype, msd, higgs
n_rounds=$2 # 1, 5, 10
cpus="0-95"

ratio=1e-03    # 1e-02, 1e-03
n_trees=10  # 1, 10, 30, 100

# 8 16 32 64
for quan in 2 4 8 16 32; do
  # iterate n_rounds times
  subdir=quan"$quan"
  cache_sub_dir="ablation-quantization/""$subdir"
  outdir="out/""$cache_sub_dir"
  mkdir -p $outdir
  mkdir -p cache/"$cache_sub_dir"
  for i in $(seq 0 $(($n_rounds-1))); do
    taskset -c $cpus ./main conf/tree10/"$dataset"_"$ratio".conf data=./data/"$dataset".train remove_ratio="$ratio" \
      n_trees="$n_trees" n_quantized_bins=$quan \
      remain_data=./data/"$dataset".train.remain_"$ratio" delete_data=./data/"$dataset".train.delete_"$ratio" \
      save_model_name="$cache_sub_dir"/"$dataset"_tree"$n_trees"_original_"$ratio"_"$i" enable_delta=true seed="$i" > \
      $outdir/"$dataset"_deltaboost_"$ratio"_"$i".out
  done
  wait

  for i in $(seq 0 "$(($n_rounds-1))"); do
    taskset -c $cpus ./main conf/tree10/"$dataset"_"$ratio".conf data=./data/"$dataset".train.remain_"$ratio" remove_ratio="$ratio" \
      n_trees="$n_trees" perform_remove=false n_quantized_bins=$quan \
      remain_data=./data/"$dataset".train.remain_"$ratio" delete_data=./data/"$dataset".train.delete_"$ratio" \
      save_model_name="$cache_sub_dir"/"$dataset"_tree"$n_trees"_retrain_"$ratio"_"$i" enable_delta=true seed="$i"   > \
      $outdir/"$dataset"_deltaboost_"$ratio"_retrain_"$i".out
  done
  wait
done