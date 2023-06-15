//
// Created by liqinbin on 12/17/20.
//
#include <iostream>
#include <fstream>
#include "FedTree/booster.h"

void Booster::init(DataSet &dataSet, const GBDTParam &param, bool get_cut_points) {

    this->param = param;

    fbuilder.reset(new HistTreeBuilder);
    if(get_cut_points)
        fbuilder->init(dataSet, param);
    else {
        fbuilder->init_nocutpoints(dataSet, param);
    }
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default")
        metric.reset(Metric::create(obj->default_metric_name()));
    else
        metric.reset(Metric::create(param.metric));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}

SyncArray<GHPair> Booster::get_gradients() {
    SyncArray<GHPair> gh;
    gh.resize(gradients.size());
    gh.copy_from(gradients);
    return gh;
}

void Booster::set_gradients(SyncArray<GHPair> &gh) {
    gradients.resize(gh.size());
    gradients.copy_from(gh);
}

void Booster::add_noise_to_gradients(float variance) {
    auto gradients_data = gradients.host_data();
    for (int i = 0; i < gradients.size(); i++) {
        DPnoises<float_type>::add_gaussian_noise(gradients_data[i].g, variance);
        DPnoises<float_type>::add_gaussian_noise(gradients_data[i].h, variance);
    }
}

void Booster::update_gradients() {
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
}

void Booster::boost(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);

    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);

    PERFORMANCE_CHECKPOINT(timerObj);

    boosted_model.push_back(fbuilder->build_approximate(gradients));

    PERFORMANCE_CHECKPOINT(timerObj);

    std::ofstream myfile;
    myfile.open ("data-base.txt", std::ios_base::app);
    myfile << fbuilder->get_y_predict() << "\n";
    myfile.close();
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}

void Booster::boost_without_prediction(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);

    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);

    PERFORMANCE_CHECKPOINT(timerObj);

    boosted_model.push_back(fbuilder->build_approximate(gradients, false));

    PERFORMANCE_CHECKPOINT(timerObj);

    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}