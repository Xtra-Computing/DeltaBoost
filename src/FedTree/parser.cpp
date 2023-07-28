//
// Created by liqinbin on 10/14/20.
// Edit by Tianyuan Fu on 10/19/2020
// Referring to the parser of ThunderGBM:
// https://github.com/Xtra-Computing/thundergbm/blob/master/src/thundergbm/parser.cpp
//

#include <FedTree/FL/FLparam.h>
#include <FedTree/parser.h>
#include <FedTree/dataset.h>
#include <FedTree/Tree/tree.h>

#include <memory>
#include "FedTree/Tree/deltaboost.h"

#include "boost/serialization/string.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/archive_exception.hpp"
#include "boost/json/src.hpp"
#include "boost/json.hpp"

using namespace std;

//TODO: code clean on compare() and atoi()
void Parser::parse_param(FLParam &fl_param, int argc, char **argv) {
    // setup default value
    fl_param.n_parties = 2; // TODO: validate the default fl values
    fl_param.mode = "horizontal";
    fl_param.partition_mode = fl_param.mode;
    fl_param.privacy_tech = "he";
    fl_param.partition= true;
    fl_param.alpha = 100;
    fl_param.n_hori = 2;
    fl_param.n_verti = 2;

    fl_param.propose_split = "server";
    fl_param.merge_histogram = "server";
    fl_param.privacy_budget = 10;
    fl_param.variance = 200;

    GBDTParam *gbdt_param = &fl_param.gbdt_param;

    gbdt_param->depth = 6;
    gbdt_param->n_trees = 40;
    gbdt_param->n_device = 1;
    gbdt_param->min_child_weight = 1;
    gbdt_param->lambda = 1;
    gbdt_param->gamma = 1;
    gbdt_param->rt_eps = 1e-6;
    gbdt_param->max_num_bin = 255;
    gbdt_param->verbose = 1;
    gbdt_param->profiling = false;
    gbdt_param->column_sampling_rate = 1;
    gbdt_param->bagging = false;
    gbdt_param->n_parallel_trees = 1;
    gbdt_param->learning_rate = 1;
    gbdt_param->objective = "reg:linear";
    gbdt_param->num_class = 1;
    gbdt_param->path = "../dataset/test_dataset.txt";
    gbdt_param->tree_method = "hist";
    gbdt_param->tree_per_round = 1; // # tree of each round, depends on # class
    gbdt_param->metric = "default";
    gbdt_param->delete_data_path = "";
    gbdt_param->remain_data_path = "";
    gbdt_param->save_model_name = "";
    gbdt_param->reorder_label = false;

    DeltaBoostParam *deltaboost_param = &fl_param.deltaboost_param;
    deltaboost_param->enable_delta = "false";
    deltaboost_param->remove_ratio = 0.0;
    deltaboost_param->dataset_name = "";
    deltaboost_param->n_used_trees = 0;
    deltaboost_param->max_bin_size = 100;
    deltaboost_param->gain_alpha = 0.0;
    deltaboost_param->nbr_size = 1;
    deltaboost_param->delta_gain_eps_feature = 0.0;
    deltaboost_param->delta_gain_eps_sn = 0.0;
    deltaboost_param->hash_sampling_round = 1;
    deltaboost_param->perform_remove = true;
    deltaboost_param->min_gain;

    if (argc < 2) {
        printf("Usage: <config>\n");
        exit(0);
    }

    //parsing parameter values from configuration file or command line
    auto parse_value = [&](const char *name_val) {
        char name[256], val[256];
        name[0] = '\0', val[0] = '\0';
        if (sscanf(name_val, "%[^=]=%s", name, val) == 2) {
            string str_name(name);

            // FL params
            if ((str_name.compare("n_parties") == 0) || (str_name.compare("num_parties") == 0) ||
                (str_name.compare("n_clients") == 0) || (str_name.compare("num_clients") == 0) ||
                (str_name.compare("n_devices") == 0) || (str_name.compare("num_devices") == 0))
                fl_param.n_parties = atoi(val);
            else if (str_name.compare("mode") == 0)
                fl_param.mode = val;
            else if ((str_name.compare("privacy") == 0) || (str_name.compare("privacy_tech") == 0) || (str_name.compare("privacy_method") == 0))
                fl_param.privacy_tech = val;
            else if (str_name.compare("partition") == 0)
                fl_param.partition = atoi(val);
            else if (str_name.compare("partition_mode") == 0)
                fl_param.partition_mode = val;
            else if (str_name.compare("alpha") == 0)
                fl_param.alpha = atof(val);
            else if (str_name.compare("n_hori") == 0)
                fl_param.n_hori = atoi(val);
            else if (str_name.compare("n_verti") == 0)
                fl_param.n_verti = atoi(val);
            else if (str_name.compare("privacy_budget") == 0)
                fl_param.privacy_budget = atof(val);
            else if (str_name.compare("merge_histogram") == 0)
                fl_param.merge_histogram = val;
            else if (str_name.compare("propose_split") == 0)
                fl_param.propose_split = val;
            // GBDT params
            else if ((str_name.compare("max_depth") == 0) || (str_name.compare("depth") == 0))
                gbdt_param->depth = atoi(val);
            else if ((str_name.compare("num_round") == 0) || (str_name.compare("n_trees") == 0))
                gbdt_param->n_trees = atoi(val);
            else if (str_name.compare("n_gpus") == 0)
                gbdt_param->n_device = atoi(val);
            else if ((str_name.compare("verbosity") == 0) || (str_name.compare("verbose") == 0))
                gbdt_param->verbose = atoi(val);
            else if (str_name.compare("profiling") == 0)
                gbdt_param->profiling = atoi(val);
            else if (str_name.compare("data") == 0)
                gbdt_param->path = val;
            else if (str_name.compare("test_data") == 0)
                gbdt_param->test_path = val;
            else if (str_name.compare("remain_data") == 0)
                gbdt_param->remain_data_path = val;
            else if (str_name.compare("delete_data") == 0)
                gbdt_param->delete_data_path = val;
            else if (str_name.compare("save_model_name") == 0)
                gbdt_param->save_model_name = val;
            else if ((str_name.compare("max_bin") == 0) || (str_name.compare("max_num_bin") == 0))
                gbdt_param->max_num_bin = atoi(val);
            else if ((str_name.compare("colsample") == 0) || (str_name.compare("column_sampling_rate") == 0))
                gbdt_param->column_sampling_rate = atof(val);
            else if (str_name.compare("bagging") == 0)
                gbdt_param->bagging = atoi(val);
            else if ((str_name.compare("num_parallel_tree") == 0) || (str_name.compare("n_parallel_trees") == 0))
                gbdt_param->n_parallel_trees = atoi(val);
            else if (str_name.compare("eta") == 0 || str_name.compare("learning_rate") == 0) {
                gbdt_param->learning_rate = atof(val);
            }
            else if (str_name.compare("objective") == 0)
                gbdt_param->objective = val;
            else if (str_name.compare("num_class") == 0)
                gbdt_param->num_class = atoi(val);
            else if (str_name.compare("min_child_weight") == 0)
                gbdt_param->min_child_weight = atoi(val);
            else if (str_name.compare("lambda") == 0 || str_name.compare("lambda_tgbm") == 0 || str_name.compare("reg_lambda") == 0)
                gbdt_param->lambda = atof(val);
            else if (str_name.compare("gamma") == 0 || str_name.compare("min_split_loss") == 0)
                gbdt_param->gamma = atof(val);
            else if (str_name.compare("tree_method") == 0)
                gbdt_param->tree_method = val;
            else if (str_name.compare("metric") == 0)
                gbdt_param->metric = val;
            else if (str_name.compare("reorder_label") == 0)
                gbdt_param->reorder_label = val;

            else if (str_name.compare("enable_delta") == 0)
                deltaboost_param->enable_delta = (strcasecmp("true", val) == 0);
            else if (str_name.compare("remove_ratio") == 0)
                deltaboost_param->remove_ratio = atof(val);
            else if (str_name.compare("min_diff_gain") == 0)
                deltaboost_param->min_diff_gain = atof(val);
            else if (str_name.compare("max_range_gain") == 0)
                deltaboost_param->max_range_gain = atof(val);
            else if (str_name.compare("dataset_name") == 0)
                deltaboost_param->dataset_name = val;
            else if (str_name.compare("n_used_trees") == 0)
                deltaboost_param->n_used_trees = atoi(val);
            else if (str_name.compare("max_bin_size") == 0)
                deltaboost_param->max_bin_size = atoi(val);
            else if (str_name.compare("gain_alpha") == 0)
                deltaboost_param->gain_alpha = atof(val);
            else if (str_name.compare("nbr_size") == 0)
                deltaboost_param->nbr_size = atoi(val);
            else if (str_name.compare("delta_gain_eps_feature") == 0)
                deltaboost_param->delta_gain_eps_feature = atof(val);
            else if (str_name.compare("delta_gain_eps_sn") == 0)
                deltaboost_param->delta_gain_eps_sn = atof(val);
            else if (str_name.compare("hash_sampling_round") == 0)
                deltaboost_param->hash_sampling_round = atoi(val);
            else if (str_name.compare("perform_remove") == 0)
                deltaboost_param->perform_remove = (strcasecmp("true", val) == 0);
            else if (str_name.compare("n_quantized_bins") == 0)
                deltaboost_param->n_quantize_bins = atoi(val);
            else if (str_name.compare("seed") == 0)
                deltaboost_param->seed = (size_t) atoi(val);
            else if (str_name.compare("min_gain") == 0)
                deltaboost_param->min_gain = atof(val);
            else
                LOG(WARNING) << "\"" << name << "\" is unknown option!";
        } else {
            string str_name(name);
            if (str_name == "-help") {
                printf("please refer to \"docs/parameters.md\" in the GitHub repository for more information about setting the options\n");
                exit(0);
            }
        }
    };

    //read configuration file
    std::ifstream conf_file(argv[1]);
    if (conf_file.fail()) {
        LOG(FATAL) << "File \"" << argv[1] << "\" does not exist.";
    }

    std::string line;
    while (std::getline(conf_file, line))
    {
        //LOG(INFO) << line;
        parse_value(line.c_str());
    }

    //TODO: confirm handling spaces around "="
    for (int i = 0; i < argc; ++i) {
        parse_value(argv[i]);
    }//end parsing parameters

    if (deltaboost_param->enable_delta) {
        // copy gbdt params into deltaboost params
        fl_param.deltaboost_param = DeltaBoostParam(gbdt_param, deltaboost_param);
    }
}

// TODO: implement Tree and DataSet; check data structure compatibility
void Parser::load_model(const string& model_path, GBDTParam &model_param, vector<vector<Tree>> &boosted_model, DataSet & dataset) {
    ifstream ifs(model_path, ios::binary);
    CHECK_EQ(ifs.is_open(), true);
    int length;
    ifs.read((char*)&length, sizeof(length));
    char * temp = new char[length+1];
    temp[length] = '\0';
    // read param.objective
    ifs.read(temp, length);
    string str(temp);
    model_param.objective = str;
    ifs.read((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
    ifs.read((char*)&model_param.num_class, sizeof(model_param.num_class));
    ifs.read((char*)&model_param.n_trees, sizeof(model_param.n_trees));
    int label_size;
    ifs.read((char*)&label_size, sizeof(label_size));
    float_type f;
    dataset.label.clear();
    for (int i = 0; i < label_size; ++i) {
        ifs.read((char*)&f, sizeof(float_type));
        dataset.label.push_back(f);
    }
    int boosted_model_size;
    ifs.read((char*)&boosted_model_size, sizeof(boosted_model_size));
    Tree t;
    vector<Tree> v;
    for (int i = 0; i < boosted_model_size; ++i) {
        int boost_model_i_size;
        ifs.read((char*)&boost_model_i_size, sizeof(boost_model_i_size));
        for (int j = 0; j < boost_model_i_size; ++j) {
            size_t syn_node_size;
            ifs.read((char*)&syn_node_size, sizeof(syn_node_size));
            SyncArray<Tree::TreeNode> tmp(syn_node_size);
            ifs.read((char*)tmp.host_data(), sizeof(Tree::TreeNode) * syn_node_size);
            t.nodes.resize(tmp.size());
            t.nodes.copy_from(tmp);
            v.push_back(t);
        }
        boosted_model.push_back(v);
        v.clear();
    }
    ifs.close();
}



void Parser::save_model(const string& model_path, GBDTParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataset) {
    ofstream out_model_file(model_path, ios::binary);
    CHECK_EQ(out_model_file.is_open(), true);
    int length = model_param.objective.length();
    out_model_file.write((char*)&length, sizeof(length));
    out_model_file.write(model_param.objective.c_str(), model_param.objective.length());
    out_model_file.write((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
    out_model_file.write((char*)&model_param.num_class, sizeof(model_param.num_class));
    out_model_file.write((char*)&model_param.n_trees, sizeof(model_param.n_trees));
    int label_size = dataset.label.size();
    out_model_file.write((char*)&label_size, sizeof(label_size));
    out_model_file.write((char*)&dataset.label[0], dataset.label.size() * sizeof(float_type));
    int boosted_model_size = boosted_model.size();
    out_model_file.write((char*)&boosted_model_size, sizeof(boosted_model_size));
    for(int j = 0; j < boosted_model.size(); ++j) {
        int boosted_model_j_size = boosted_model[j].size();
        out_model_file.write((char*)&boosted_model_j_size, sizeof(boosted_model_j_size));
        for (int i = 0; i < boosted_model_j_size; ++i) {
            size_t syn_node_size = boosted_model[j][i].nodes.size();
            out_model_file.write((char*)&syn_node_size, sizeof(syn_node_size));
            out_model_file.write((char*)boosted_model[j][i].nodes.host_data(), syn_node_size * sizeof(Tree::TreeNode));
        }
    }
    out_model_file.close();
}

void Parser::load_model(const string &model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet &dataset) {
    LOG(INFO) << "Loading from " << model_path;
    auto &boosted_model = model.trees;
    boosted_model.clear();
    ifstream ifs(model_path, ios::binary);
    CHECK_EQ(ifs.is_open(), true);
    int length;
    ifs.read((char*)&length, sizeof(length));
    char * temp = new char[length+1];
    temp[length] = '\0';
    // read param.objective
    ifs.read(temp, length);
    string str(temp);
    model_param.objective = str;
    ifs.read((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
    ifs.read((char*)&model_param.num_class, sizeof(model_param.num_class));
    ifs.read((char*)&model_param.n_trees, sizeof(model_param.n_trees));
    int label_size;
    ifs.read((char*)&label_size, sizeof(label_size));
    float_type f;
    dataset.label.clear();
    for (int i = 0; i < label_size; ++i) {
        ifs.read((char*)&f, sizeof(float_type));
        dataset.label.push_back(f);
    }

    // write model
    boost::archive::text_iarchive ia(ifs);
    ia >> model;
    ifs.close();

    LOG(INFO) << "Loaded.";
}

void Parser::save_model(const string &model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet &dataSet) {
    const vector<vector<DeltaTree>> &boosted_model = model.trees;
    ofstream out_model_file(model_path, ios::binary);
    CHECK_EQ(out_model_file.is_open(), true);
    int length = model_param.objective.length();
    out_model_file.write((char*)&length, sizeof(length));
    out_model_file.write(model_param.objective.c_str(), model_param.objective.length());
    out_model_file.write((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
    out_model_file.write((char*)&model_param.num_class, sizeof(model_param.num_class));
    out_model_file.write((char*)&model_param.n_trees, sizeof(model_param.n_trees));
    int label_size = dataSet.label.size();
    out_model_file.write((char*)&label_size, sizeof(label_size));
    out_model_file.write((char*)&dataSet.label[0], dataSet.label.size() * sizeof(float_type));

    // save model
    boost::archive::text_oarchive oa(out_model_file);
    oa << model;
    out_model_file.close();

    LOG(INFO) << "saved to " << model_path;
}

void Parser::save_model_to_json(const string &model_path, DeltaBoostParam &model_param, DeltaBoost &model,
                                DataSet &dataSet) {
    json::object v;
    v["objective"] = model_param.objective;
    v["learning_rate"] = model_param.learning_rate;
    v["num_class"] = model_param.num_class;
    v["n_trees"] = model_param.n_trees;
    v["label_size"] = dataSet.label.size();

    assert(dataSet.label.empty() == false);

    if constexpr(std::is_same_v<float_type, double>) {
        v["labels"] = json::value_from(dataSet.label);
    } else {
        vector<double> labels(dataSet.label.size());
        for (int i = 0; i < dataSet.label.size(); ++i) {
            labels[i] = (double) dataSet.label[i];
        }
        v["labels"] = json::value_from(labels);
    }


    v["deltaboost"] = json::value_from(model);

    ofstream out_model_file(model_path);
    CHECK_EQ(out_model_file.is_open(), true);

    out_model_file << boost::json::serialize(v) << endl;
    out_model_file.close();

    LOG(INFO) << "saved to " << model_path;
}

void Parser::save_model_to_json(const string &model_path, GBDTParam &model_param, GBDT &model, DataSet &dataSet) {
    json::object v;
    v["objective"] = model_param.objective;
    v["learning_rate"] = model_param.learning_rate;
    v["num_class"] = model_param.num_class;
    v["n_trees"] = model_param.n_trees;
    v["label_size"] = dataSet.label.size();

    assert(dataSet.label.empty() == false);

    if constexpr(std::is_same_v<float_type, double>) {
        v["labels"] = json::value_from(dataSet.label);
    } else {
        vector<double> labels(dataSet.label.size());
        for (int i = 0; i < dataSet.label.size(); ++i) {
            labels[i] = (double) dataSet.label[i];
        }
        v["labels"] = json::value_from(labels);
    }

    v["gbdt"] = json::value_from(model);

    ofstream out_model_file(model_path);
    CHECK_EQ(out_model_file.is_open(), true);

    out_model_file << boost::json::serialize(v) << endl;
    out_model_file.close();

    LOG(INFO) << "saved to " << model_path;
}

void Parser::save_scores_to_csv(const string &score_path, const vector<float_type> &scores,
                                const vector<float_type> &labels) {
    ofstream out_score_file(score_path);
    CHECK_EQ(out_score_file.is_open(), true);
    assert(scores.size() == labels.size());
    for (int i = 0; i < scores.size() - 1; ++i) {
        out_score_file << scores[i] << "," << labels[i] << endl;
    }
    out_score_file << scores[scores.size() - 1] << "," << labels[labels.size() - 1];
    out_score_file.close();
    LOG(INFO) << "saved to " << score_path;
}




//void Parser::load_model(const string& model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet & dataset) {
//    LOG(INFO) << "Loading from " << model_path;
//
//    auto &boosted_model = model.trees;
//    boosted_model.clear();
//    ifstream ifs(model_path, ios::binary);
//    CHECK_EQ(ifs.is_open(), true);
//    int length;
//    ifs.read((char*)&length, sizeof(length));
//    char * temp = new char[length+1];
//    temp[length] = '\0';
//    // read param.objective
//    ifs.read(temp, length);
//    string str(temp);
//    model_param.objective = str;
//    ifs.read((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
//    ifs.read((char*)&model_param.num_class, sizeof(model_param.num_class));
//    ifs.read((char*)&model_param.n_trees, sizeof(model_param.n_trees));
//    int label_size;
//    ifs.read((char*)&label_size, sizeof(label_size));
//    float_type f;
//    dataset.label.clear();
//    for (int i = 0; i < label_size; ++i) {
//        ifs.read((char*)&f, sizeof(float_type));
//        dataset.label.push_back(f);
//    }
//    int boosted_model_size;
//    ifs.read((char*)&boosted_model_size, sizeof(boosted_model_size));
//    boosted_model.resize(boosted_model_size);
//    for (int i = 0; i < boosted_model_size; ++i) {
//        int boost_model_i_size;
//        ifs.read((char*)&boost_model_i_size, sizeof(boost_model_i_size));
//        vector<DeltaTree> v(boost_model_i_size);
//        for (int j = 0; j < boost_model_i_size; ++j) {
//            DeltaTree tree;
//            size_t syn_node_size;
//            ifs.read((char*)&syn_node_size, sizeof(syn_node_size));
//            tree.nodes.resize(syn_node_size);
//
//            for (int k = 0; k < syn_node_size; ++k) {
//                DeltaTree::DeltaNode tmp_node;
//                size_t potential_nodes_indices_size;
//                ifs.read((char*)(&potential_nodes_indices_size), sizeof(size_t));
//                vector<int> potential_node_indices(potential_nodes_indices_size);
//                for (int t = 0, idx = -1; t < potential_nodes_indices_size; ++t) {
//                    ifs >> idx;
//                    potential_node_indices.push_back(idx);
//                }
//                ifs.read((char*)(&tmp_node), sizeof(DeltaTree::DeltaNode));
//
//                if (!tree.nodes[0].potential_nodes_indices.empty() && tree.nodes[0].potential_nodes_indices[2] != 2) {
//                    LOG(INFO);
//                }
//                tree.nodes[k].potential_nodes_indices = potential_node_indices;
//                tree.nodes[k] = tmp_node;
//            }
//            v[j] = tree;
//        }
//        boosted_model[i] = v;
//    }
//
//    size_t num_iters = -1, num_samples = -1;
//    ifs.read((char*)(&num_iters), sizeof(size_t));
//    auto res = ifs.rdstate();
//    model.gh_pairs_per_sample.clear();
//    for (int i = 0; i < num_iters; ++i) {
//        ifs.read((char*)&num_samples, sizeof(size_t));
//        vector<GHPair> gh_pairs_iter_i;
//        for (int j = 0; j < num_samples; ++j) {
//            GHPair gh_pair;
//            ifs.read((char*)&gh_pair.g, sizeof(float_type));
//            ifs.read((char*)&gh_pair.h, sizeof(float_type));
//            gh_pairs_iter_i.emplace_back(gh_pair);
//        }
//        model.gh_pairs_per_sample.emplace_back(gh_pairs_iter_i);
//    }
//
//    ifs.close();
//
//    LOG(INFO) << "Loaded.";
//}
//
//
//void Parser::save_model(const string& model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet &dataset) {
//
//    const vector<vector<DeltaTree>> &boosted_model = model.trees;
//    ofstream out_model_file(model_path, ios::binary);
//    CHECK_EQ(out_model_file.is_open(), true);
//    int length = model_param.objective.length();
//    out_model_file.write((char*)&length, sizeof(length));
//    out_model_file.write(model_param.objective.c_str(), model_param.objective.length());
//    out_model_file.write((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
//    out_model_file.write((char*)&model_param.num_class, sizeof(model_param.num_class));
//    out_model_file.write((char*)&model_param.n_trees, sizeof(model_param.n_trees));
//    int label_size = dataset.label.size();
//    out_model_file.write((char*)&label_size, sizeof(label_size));
//    out_model_file.write((char*)&dataset.label[0], dataset.label.size() * sizeof(float_type));
//    int boosted_model_size = boosted_model.size();
//    out_model_file.write((char*)&boosted_model_size, sizeof(boosted_model_size));
//    for(int j = 0; j < boosted_model.size(); ++j) {
//        int boosted_model_j_size = boosted_model[j].size();
//        out_model_file.write((char*)&boosted_model_j_size, sizeof(boosted_model_j_size));
//        for (int i = 0; i < boosted_model_j_size; ++i) {
//            size_t syn_node_size = boosted_model[j][i].nodes.size();
//            out_model_file.write((char*)&syn_node_size, sizeof(syn_node_size));
//            for (int k = 0; k < syn_node_size; ++k) {
//                size_t potential_nodes_indices_size = boosted_model[j][i].nodes[k].potential_nodes_indices.size();
//                out_model_file.write((char*)(&potential_nodes_indices_size), sizeof(size_t));
//                out_model_file.write((char*)boosted_model[j][i].nodes[k].potential_nodes_indices.data(), potential_nodes_indices_size * sizeof(int));
//                out_model_file.write((char*)(&boosted_model[j][i].nodes[k]), sizeof(DeltaTree::DeltaNode));
//            }
//        }
//    }
//
//    const vector<vector<GHPair>> &gh_pairs_per_sample = model.gh_pairs_per_sample;
//    size_t num_iters = gh_pairs_per_sample.size();
//    out_model_file.write((char*)(&num_iters), sizeof(size_t));
//    auto res = out_model_file.rdstate();
//    for (int i = 0; i < gh_pairs_per_sample.size(); ++i) {
//        size_t num_samples = gh_pairs_per_sample[i].size();
//        out_model_file.write((char*)(&num_samples), sizeof(size_t));
//        for (int j = 0; j < gh_pairs_per_sample[i].size(); ++j) {
//            out_model_file.write((char*)(&gh_pairs_per_sample[i][j].g), sizeof(float_type));
//            out_model_file.write((char*)(&gh_pairs_per_sample[i][j].h), sizeof(float_type));
//        }
//    }
//
//    out_model_file.close();
//
//    LOG(INFO) << "saved to " << model_path;
//}
