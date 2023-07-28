n_trees=$1  # 1, 10, 30, 100
dataset=$2  # cadata, codrna, gisette, covtype, msd, higgs
cpus="0-62"

: "${1?"Number of trees unset."}"
: "${1?"Dataset unset."}"
: "${1?"Ratio unset."}"
: "${1?"Number of rounds unset."}"

subdir=tree"$n_trees"
outdir="out/remove_test/$subdir/"
mkdir -p $outdir


# run one time because GBDT is determinstic
taskset -c $cpus ./main conf/"$subdir"/"$dataset"_1e-02.conf save_model_name="$dataset"_tree"$n_trees"_gbdt data=./data/"$dataset".train enable_delta=false n_trees=$n_trees > \
          $outdir/"$dataset"_gbdt.out



