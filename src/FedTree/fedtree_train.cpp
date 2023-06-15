//
// Created by liqinbin on 10/13/20.
//

#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/deltaboost.h"


#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif


int main(int argc, char** argv){
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

    FLParam fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);
    GBDTParam &model_param = fl_param.gbdt_param;

    if(model_param.verbose == 0) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    }
    else if (model_param.verbose == 1) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }

    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }

    int n_parties = fl_param.n_parties;
    vector<DataSet> train_subsets(n_parties);
    vector<DataSet> test_subsets(n_parties);
    vector<DataSet> subsets(n_parties);
    vector<SyncArray<bool>> feature_map(n_parties);
    std::map<int, vector<int>> batch_idxs;
    DataSet dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    dataset.load_from_csv(model_param.path, fl_param);

    if (!dataset.has_csc) {
        dataset.csr_to_csc();
    }

    DataSet test_dataset;
    if (use_global_test_set)
        test_dataset.load_from_csv(model_param.test_path, fl_param);

    GBDTParam &param = fl_param.gbdt_param;

    LOG(INFO) << "start training";
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        std::cout<<"FedTree only supports histogram-based training yet";
        exit(1);
    }
    std::vector<float_type> scores;

    if(fl_param.mode == "centralized"){

        DataSet remain_dataset, delete_dataset;
        bool test_on_remain = !fl_param.gbdt_param.remain_data_path.empty();
        bool test_on_delete = !fl_param.gbdt_param.delete_data_path.empty();
        if (test_on_remain) {
            remain_dataset.load_from_csv(fl_param.gbdt_param.remain_data_path, fl_param);
        }
        if (test_on_delete) {
            delete_dataset.load_from_csv(fl_param.gbdt_param.delete_data_path, fl_param);
        }

        if (fl_param.deltaboost_param.enable_delta) {
            auto deltaboost = std::unique_ptr<DeltaBoost>(new DeltaBoost());
            float_type score;
            string model_path = string_format("cache/%s_deltaboost.model",
                                              fl_param.deltaboost_param.save_model_name.c_str());
            deltaboost->train(fl_param.deltaboost_param, dataset);      // train
//            parser.save_model(model_path, fl_param.deltaboost_param, *deltaboost, dataset);     // save

//            auto deltaboost_load = std::unique_ptr<DeltaBoost>(new DeltaBoost());
//            parser.load_model(model_path, fl_param.deltaboost_param, *deltaboost, dataset);     // load

            deltaboost->trim_unused_members_();


            string model_path_json = string_format("cache/%s_deltaboost.json",
                                              fl_param.deltaboost_param.save_model_name.c_str());
            parser.save_model_to_json(model_path_json, fl_param.deltaboost_param, *deltaboost, dataset);

            LOG(INFO) << "On test dataset";
            vector<float_type> test_scores;
            deltaboost->predict_score(fl_param.deltaboost_param, test_dataset, test_scores,
                                      fl_param.deltaboost_param.n_used_trees);
            string test_score_path = string_format("cache/%s_deltaboost_score_test.csv",
                                                   fl_param.deltaboost_param.save_model_name.c_str());
            parser.save_scores_to_csv(test_score_path, test_scores, test_dataset.y);

            if (test_on_delete) {
                LOG(INFO) << "On deleted dataset";
                deltaboost->predict_score(fl_param.deltaboost_param, delete_dataset,
                                          fl_param.deltaboost_param.n_used_trees);
            }
            if (test_on_remain) {
                LOG(INFO) << "On remained dataset";
                deltaboost->predict_score(fl_param.deltaboost_param, remain_dataset,
                                          fl_param.deltaboost_param.n_used_trees);
            }

            if (fl_param.deltaboost_param.perform_remove) {
                // start removal
                int num_removals = static_cast<int>(fl_param.deltaboost_param.remove_ratio * dataset.n_instances());
                LOG(INFO) << num_removals << " samples to be removed from model";
                vector<int> removing_indices(static_cast<int>(fl_param.deltaboost_param.remove_ratio * dataset.n_instances()));
                std::iota(removing_indices.begin(), removing_indices.end(), 0);

                typedef std::chrono::high_resolution_clock timer;
                auto start_rm = timer::now();

//                deltaboost->remove_samples(fl_param.deltaboost_param, dataset, removing_indices);
                ////////////////////////////////////////// start
                auto &param = fl_param.deltaboost_param;
                const auto &sample_indices = removing_indices;

//                typedef std::chrono::high_resolution_clock timer;
                auto start_time = timer::now();
                auto end_time = timer::now();
                std::chrono::duration<double> duration = end_time - start_time;

                LOG(INFO) << "start removing samples";

                start_time = timer::now();

                SyncArray<float_type> y = SyncArray<float_type>(dataset.n_instances());
                y.copy_from(dataset.y.data(), dataset.n_instances());
                std::unique_ptr<ObjectiveFunction> obj(ObjectiveFunction::create(param.objective));
                obj->configure(param, dataset);     // slicing param

                end_time = timer::now();
                duration = end_time - start_time;
                LOG(INFO) << "Copy y time: " << duration.count();

                LOG(INFO) << "Preparing for deletion";

                DeltaBoostRemover deltaboost_remover;
                if (param.hash_sampling_round > 1) {
                    deltaboost_remover = DeltaBoostRemover(&dataset, &(deltaboost->trees), deltaboost->is_subset_indices_in_tree, obj.get(), param);
                } else {
                    start_time = timer::now();

                    deltaboost_remover = DeltaBoostRemover(&dataset, &(deltaboost->trees), obj.get(), param);

                    end_time = timer::now();
                    duration = end_time - start_time;
                    LOG(DEBUG) << "[Removing time] Step 0 (out) = " << duration.count();
                }

                deltaboost_remover.n_all_instances = dataset.n_instances();

//    deltaboost_remover.get_info_by_prediction(gh_pairs_per_sample);
                deltaboost_remover.get_info(deltaboost->gh_pairs_per_sample, deltaboost->ins2node_indices_per_tree);

                LOG(INFO) << "Deleting " << param.n_used_trees << " trees";

#pragma omp parallel for
                for (int i = 0; i < param.n_used_trees; ++i) {

                    DeltaTreeRemover& tree_remover = deltaboost_remover.tree_removers[i];
                    vector<bool> is_iid_removed = indices_to_hash_table(sample_indices, dataset.n_instances());
                    tree_remover.is_iid_removed = is_iid_removed;
                    const std::vector<GHPair>& gh_pairs = tree_remover.gh_pairs;
                    vector<int> trained_sample_indices;
                    if (param.hash_sampling_round > 1) {
                        std::copy_if(sample_indices.begin(), sample_indices.end(), std::back_inserter(trained_sample_indices), [&](int idx){
                            return deltaboost->is_subset_indices_in_tree[i][idx];
                        });
                    } else {
                        trained_sample_indices = sample_indices;
                    }

                    tree_remover.remove_samples_by_indices(trained_sample_indices);
                    tree_remover.prune();
                }

                end_time = timer::now();
                duration = end_time - start_time;
                LOG(INFO) << "Removing time in function = " << duration.count();

                auto stop_rm = timer::now();
                std::chrono::duration<float> removing_time = stop_rm - start_rm;
                LOG(INFO) << "removing time = " << removing_time.count();


                //////////////////////// end

                LOG(INFO) << "Predict after removals";
                LOG(INFO) << "On test dataset";
                vector<float_type> deleted_test_scores;
                deltaboost->predict_score(fl_param.deltaboost_param, test_dataset, deleted_test_scores,
                                          fl_param.deltaboost_param.n_used_trees);
                string deleted_test_score_path = string_format("cache/%s_deleted_score_test.csv",
                                                       fl_param.deltaboost_param.save_model_name.c_str());
                parser.save_scores_to_csv(deleted_test_score_path, deleted_test_scores, test_dataset.y);

                if (test_on_delete) {
                    LOG(INFO) << "On deleted dataset";
                    deltaboost->predict_score(fl_param.deltaboost_param, delete_dataset,
                                              fl_param.deltaboost_param.n_used_trees);
                }
                if (test_on_remain) {
                    LOG(INFO) << "On remained dataset";
                    deltaboost->predict_score(fl_param.deltaboost_param, remain_dataset,
                                              fl_param.deltaboost_param.n_used_trees);
                }

                string model_path_json_delete = string_format("cache/%s_deleted.json",
                                                              fl_param.deltaboost_param.save_model_name.c_str());
                parser.save_model_to_json(model_path_json_delete, fl_param.deltaboost_param, *deltaboost, dataset);
            }
        } else {
            auto gbdt = std::unique_ptr<GBDT>(new GBDT());
            gbdt->train(fl_param.gbdt_param, dataset);

            // save model
            string model_path = string_format("cache/%s_gbdt.model",
                                              fl_param.gbdt_param.save_model_name.c_str());
            parser.save_model(model_path, fl_param.gbdt_param, gbdt->trees, dataset);

            LOG(INFO) << "On test dataset";
            gbdt->predict_score(fl_param.gbdt_param, test_dataset);
            if (test_on_delete) {
                LOG(INFO) << "On deleted dataset";
                gbdt->predict_score(fl_param.gbdt_param, delete_dataset);
            }
            if (test_on_remain) {
                LOG(INFO) << "On remained dataset";
                gbdt->predict_score(fl_param.gbdt_param, remain_dataset);
            }

            string model_path_json = string_format("cache/%s_gbdt.json",
                                                   fl_param.gbdt_param.save_model_name.c_str());
            parser.save_model_to_json(model_path_json, fl_param.gbdt_param, *gbdt, dataset);
        }

    } else {
        LOG(FATAL) << "Unknown model type";
    }
//        parser.save_model("global_model", fl_param.gbdt_param, server.global_trees.trees, dataset);
//    }
    return 0;
}
