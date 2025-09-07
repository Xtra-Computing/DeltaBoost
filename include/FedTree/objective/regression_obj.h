//
// Created by Kelly Yung on 2020/11/27.
//

#ifndef FEDTREE_REGRESSION_OBJ_H
#define FEDTREE_REGRESSION_OBJ_H

#include "objective_function.h"
#include "FedTree/util/device_lambda.h"
#include "math.h"

template<template<typename> class Loss>
class RegressionObj : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
//        CHECK_EQ(y.size(), y_p.size())<<y.size() << "!=" << y_p.size();
//        CHECK_EQ(y.size(), gh_pair.size());
        auto y_data = y.host_data();
        auto y_p_data = y_p.host_data();
        auto gh_pair_data = gh_pair.host_data();
        for (int i = 0; i < y.size(); i++) {
            gh_pair_data[i] = Loss<float_type>::gradient(y_data[i], y_p_data[i]);
        };
    }

    void predict_transform(SyncArray<float_type> &y) override {
        auto y_data = y.host_data();
        for (int i = 0; i < y.size(); i++) {
            y_data[i] = Loss<float_type>::predict_transform(y_data[i]);
        };
    }

    void configure(GBDTParam param, const DataSet &dataset) override {}

    virtual ~RegressionObj() override = default;

    string default_metric_name() override {
        return "rmse";
    }
};

template<template<typename> class Loss>
class LogClsObj: public RegressionObj<Loss>{
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
        auto y_data = y.host_data();
        auto y_p_data = y_p.host_data();
        auto gh_pair_data = gh_pair.host_data();
        for (int i = 0; i < y.size(); i++){
            gh_pair_data[i] = Loss<float_type>::gradient(y_data[i], y_p_data[i]);
        }
    }
    void predict_transform(SyncArray<float_type> &y) {
        //this method transform y(#class * #instances) into y(#instances)
        auto yp_data = y.host_data();
        auto label_data = label.host_data();
        int num_class = this->num_class;
        int n_instances = y.size();
        for (int i = 0; i < n_instances; i++) {
            int max_k = (yp_data[i] > 0) ? 1 : 0;
            yp_data[i] = label_data[max_k];
        }
        SyncArray < float_type > temp_y(n_instances);
        temp_y.copy_from(y.host_data(), n_instances);
        y.resize(n_instances);
        y.copy_from(temp_y);
    }
    string default_metric_name() override{
        return "error";
    }
    void configure(GBDTParam param, const DataSet &dataset) {
        num_class = param.num_class;
        label.resize(num_class);
        if (dataset.label.size() == num_class) {
            label.copy_from(dataset.label.data(), num_class);
        } else {
            // Fallback: dataset may not contain all classes (e.g., filtered subsets).
            // Use canonical labels 0..num_class-1 to avoid crashes.
            std::vector<float_type> default_labels(num_class);
            for (int i = 0; i < num_class; ++i) default_labels[i] = static_cast<float_type>(i);
            label.copy_from(default_labels.data(), num_class);
        }
    }
protected:
    int num_class;
    SyncArray<float_type> label;
};

template<typename T>
struct SquareLoss {
    HOST_DEVICE static GHPair gradient(T y, T y_p) { return GHPair(y_p - y, 1); }
    HOST_DEVICE static T predict_transform(T x) { return x; }
};

//for probability regression
template<typename T>
struct LogisticLoss {
    HOST_DEVICE static GHPair gradient(T y, T y_p);

    HOST_DEVICE static T predict_transform(T x);
};

template<>
struct LogisticLoss<float> {
    HOST_DEVICE static GHPair gradient(float y, float y_p) {
        float p = sigmoid(y_p);
        return GHPair(p - y, fmaxf(p * (1 - p), 1e-16f));
    }

    HOST_DEVICE static float predict_transform(float y) { return sigmoid(y); }

    HOST_DEVICE static float sigmoid(float x) {return 1 / (1 + expf(-x));}
};

template<>
struct LogisticLoss<double> {
    HOST_DEVICE static GHPair gradient(double y, double y_p) {
        double p = sigmoid(y_p);
        return GHPair(p - y, fmax(p * (1 - p), 1e-16));
    }

    HOST_DEVICE static double predict_transform(double x) { return 1 / (1 + exp(-x)); }

    HOST_DEVICE static double sigmoid(double x) {return 1 / (1 + exp(-x));}
};


template<>
struct LogisticLoss<long double> {
    HOST_DEVICE static GHPair gradient(long double y, long double y_p) {
        long double p = sigmoid(y_p);
        return GHPair(p - y, std::max(p * (1 - p), (long double) 1e-16));
    }

    HOST_DEVICE static long double predict_transform(long double x) { return 1 / (1 + exp(-x)); }

    HOST_DEVICE static long double sigmoid(long double x) {return 1 / (1 + exp(-x));}
};


#endif //FEDTREE_REGRESSION_OBJ_H
