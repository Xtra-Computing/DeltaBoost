//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDT_H
#define FEDTREE_GBDT_H

#include "tree.h"
#include "FedTree/dataset.h"

class GBDT {
public:
    vector<vector<Tree>> trees;

    GBDT() = default;

    GBDT(const vector<vector<Tree>> gbdt){
        trees = gbdt;
    }

    void train(GBDTParam &param, DataSet &dataset);

    vector<float_type> predict(const GBDTParam &model_param, const DataSet &dataSet);

    vector<float_type> predict(const GBDTParam &model_param, const vector<DataSet> &dataSet);

    void predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict);

    void predict_raw_vertical(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict, std::map<int, vector<int>> &batch_idxs);

    void predict_raw_vertical(const GBDTParam &model_param, const vector<DataSet> &dataSet, SyncArray<float_type> &y_predict);

    float_type predict_score(const GBDTParam &model_param, const DataSet &dataSet);

    float_type predict_score_vertical(const GBDTParam &model_param, const DataSet &dataSet, std::map<int, vector<int>> &batch_idxs);

    float_type predict_score_vertical(const GBDTParam &model_param, const vector<DataSet> &dataSet);

    // json parser
    friend GBDT tag_invoke(json::value_to_tag<GBDT>, json::value const& v) {
        auto &o = v.as_object();

        GBDT gbdt;

        gbdt.trees = json::value_to<std::vector<std::vector<Tree>>>(v.at("trees"));

        return gbdt;
    }

    //json parser
    friend void tag_invoke(json::value_from_tag, json::value& v, GBDT const& gbdt) {
        v = json::object {
                {"trees", json::value_from(gbdt.trees)},
        };
    }

};

#endif //FEDTREE_GBDT_H
