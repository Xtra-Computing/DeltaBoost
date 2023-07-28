n_rounds=$1

bash ablation_ratio.sh msd $n_rounds
bash ablation_quantization.sh msd $n_rounds

bash ablation_bagging.sh codrna $n_rounds
bash ablation_iteration.sh codrna $n_rounds
bash ablation_nbins.sh codrna $n_rounds
bash ablation_quantization.sh codrna $n_rounds
bash ablation_ratio.sh codrna $n_rounds
bash ablation_regularization.sh codrna $n_rounds


