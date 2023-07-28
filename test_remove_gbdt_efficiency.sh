n_trees=$1  # 1, 10, 30, 100
#dataset=$2  # cadata, codrna, gisette, covtype, msd
cpus="0-95"

subdir=tree"$n_trees"
outdir="out/remove_test/$subdir/"
mkdir -p $outdir


for dataset in cadata codrna gisette covtype msd; do
  # run one time because GBDT is determinstic
  taskset -c $cpus ./main conf/"$subdir"/"$dataset"_1e-02.conf save_model_name="$dataset"_tree"$n_trees"_gbdt_1e-02 \
            data=./data/"$dataset".train.remain_1e-02 enable_delta=false n_trees=$n_trees > \
            $outdir/"$dataset"_gbdt_1e-02.out

  taskset -c $cpus ./main conf/"$subdir"/"$dataset"_1e-03.conf save_model_name="$dataset"_tree"$n_trees"_gbdt_1e-03 \
            data=./data/"$dataset".train.remain_1e-03 enable_delta=false n_trees=$n_trees > \
            $outdir/"$dataset"_gbdt_1e-03.out
done



