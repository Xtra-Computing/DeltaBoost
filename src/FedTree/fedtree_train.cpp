//
// Created by liqinbin on 10/13/20.
//

#include "FedTree/FL/FLparam.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
//#include "FedTree/Tree/gbdt.h"
#include "FedTree/Tree/deltaboost.h"


#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif


int main(int argc, char** argv){
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

/*
    //initialize parameters
    FLParam fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);

    //load dataset from file/files
    DataSet dataset;
    dataset.load_from_file(fl_param.dataset_path);

    //initialize parties and server *with the dataset*
    vector<Party> parties;
    for(i = 0; i < fl_param.n_parties; i++){
        Party party;
        parties.push_back(party);
    }
    Server server;

    //train
    FLtrainer trainer;
    model = trainer.train(parties, server, fl_param);

    //test
    Dataset test_dataset;
    test_dataset.load_from_file(fl_param.test_dataset_path);
    acc = model.predict(test_dataset);
*/

    omp_set_dynamic(0);
//    omp_set_num_threads(64);

//centralized training test
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
//    if(fl_param.mode == "centralized") {
//        DataSet dataset;
//        vector <vector<Tree>> boosted_model;
//        dataset.load_from_file(model_param.path, fl_param);
//        std::map<int, vector<int>> batch_idxs;
//        Partition partition;
//        vector<DataSet> subsets(3);
//        partition.homo_partition(dataset, 3, true, subsets, batch_idxs);
//        GBDT gbdt;
//        gbdt.train(model_param, dataset);
////       float_type score = gbdt.predict_score(model_param, dataset);
//       // LOG(INFO) << score;
//      //  parser.save_model("tgbm.model", model_param, gbdt.trees, dataset);
//    }
//    }
//    else{
    int n_parties = fl_param.n_parties;
    vector<DataSet> train_subsets(n_parties);
    vector<DataSet> test_subsets(n_parties);
    vector<DataSet> subsets(n_parties);
    vector<SyncArray<bool>> feature_map(n_parties);
    std::map<int, vector<int>> batch_idxs;
    DataSet dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    dataset.load_from_csv(model_param.path, fl_param);

    DataSet test_dataset;
    if (use_global_test_set)
        test_dataset.load_from_csv(model_param.test_path, fl_param);

//    if (ObjectiveFunction::need_group_label(param.gbdt_param.objective)) {
//        group_label();
//        param.gbdt_param.num_class = label.size();
//    }

    GBDTParam &param = fl_param.gbdt_param;

//    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos || param.metric == "error") {
//        for (int i = 0; i < n_parties; i++) {
//            train_subsets[i].group_label();
//            test_subsets[i].group_label();
//        }
//        int num_class = dataset.label.size();
//        if (param.num_class != num_class) {
//            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
//            param.num_class = num_class;
//        }
//        if(param.num_class > 2)
//            param.tree_per_rounds = param.num_class;
//    }
//    else if(param.objective.find("reg:") != std::string::npos){
//        param.num_class = 1;
//    }


    LOG(INFO) << "start training";
    FLtrainer trainer;
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
            deltaboost->train(fl_param.deltaboost_param, dataset);
//            parser.save_model(model_path, fl_param.deltaboost_param, *deltaboost, dataset);
//            parser.load_model(model_path, fl_param.deltaboost_param, *deltaboost, dataset);

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
                std::chrono::high_resolution_clock timer;
                auto start_rm = timer.now();
                int num_removals = static_cast<int>(fl_param.deltaboost_param.remove_ratio * dataset.n_instances());
                LOG(INFO) << num_removals << " samples to be removed from model";
                vector<int> removing_indices(static_cast<int>(fl_param.deltaboost_param.remove_ratio * dataset.n_instances()));
                std::iota(removing_indices.begin(), removing_indices.end(), 0);
                deltaboost->remove_samples(fl_param.deltaboost_param, dataset, removing_indices);
                auto stop_rm = timer.now();
                std::chrono::duration<float> removing_time = stop_rm - start_rm;
                LOG(INFO) << "removing time = " << removing_time.count();

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
