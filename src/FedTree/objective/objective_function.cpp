//
// Created by liqinbin on 12/15/20.
//

#include <FedTree/objective/objective_function.h>
#include "FedTree/objective/regression_obj.h"
#include "FedTree/objective/multiclass_obj.h"

ObjectiveFunction *ObjectiveFunction::create(string name) {
    if (name == "reg:linear") return new RegressionObj<SquareLoss>;
    if (name == "reg:logistic") return new RegressionObj<LogisticLoss>;
    if (name == "binary:logistic") return new LogClsObj<LogisticLoss>;
    if (name == "multi:softprob") return new SoftmaxProb;
    if (name == "multi:softmax") return new Softmax;
    LOG(FATAL) << "undefined objective " << name;
    return nullptr;
}

bool ObjectiveFunction::need_load_group_file(string name) {
    return name == "rank:ndcg" || name == "rank:pairwise";
}

bool ObjectiveFunction::need_group_label(string name) {
    return name == "multi:softprob" || name == "multi:softmax" || name == "binary:logistic";
}
