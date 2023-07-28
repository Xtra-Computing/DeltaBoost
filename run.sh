mkdir -p cache out fig

echo "Installing Python dependencies"
pip install -r requirements.txt

echo "Downloading datasets and removing samples"
bash download_datasets.sh

# build DeltaBoost
mkdir build && cd build
cmake ..
make -j
cd ..   # under root dir of DeltaBoost
ln -s build/bin/FedTree-train main

# run DeltaBoost
echo "Table 4,5"
bash test_remove_deltaboost_tree_1.sh 100  # try 100 seeds
cd python-utils
python plot_results.py -t 1
cd -

echo "Table 7"
bash test_remove_deltaboost_tree_10.sh 100  # try 100 seeds
cd python-utils
python plot_results.py -t 10
cd -

echo "Table 6"
taskset -c 0-95 python baseline.py
bash test_remove_gbdt_efficiency.sh
cd python-utils
python plot_time.py -t 10
cd -

echo "Figure 9"
python baseline.py -t 10
python baseline.py -t 100
bash test_accuracy.sh 10  # run 10 times
cd python-utils
python plot_results.py -acc -t 10   # (10 trees)
python plot_results.py -acc -t 100  # (100 trees)
cd -

echo "Figure 10,11"
bash test_all_ablation.sh 50  # run 50 times
cd python-utils
python plot_ablation.py
cd -

echo "Finished, please refer to fig/ and out/ for results."