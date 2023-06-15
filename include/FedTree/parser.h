//
// Created by liqinbin on 10/13/20.
// Edited by Tianyuan Fu on 10/19/20
//

#ifndef FEDTREE_PARSER_H
#define FEDTREE_PARSER_H
#include "FedTree/common.h"

#include <FedTree/FL/FLparam.h>
#include "dataset.h"
#include "Tree/tree.h"
#include "Tree/deltaboost.h"

// Todo: parse the parameters to FLparam. refer to ThunderGBM parser.h https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/parser.h
class Parser {
public:
    void parse_param(FLParam &fl_param, int argc, char **argv);
    void load_model(const string& model_path, GBDTParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataSet);
    void save_model(const string& model_path, GBDTParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataSet);
    void load_model(const string& model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet & dataset);
    void save_model(const string& model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet &dataSet);
    void save_model_to_json(const string& model_path, DeltaBoostParam &model_param, DeltaBoost &model, DataSet &dataSet);
    void save_model_to_json(const string& model_path, GBDTParam &model_param, GBDT &model, DataSet &dataSet);
    void save_scores_to_csv(const string& score_path, const vector<float_type> &scores, const vector<float_type> &labels);
};

#endif //FEDTREE_PARSER_H
