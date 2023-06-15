//
// Created by liqinbin on 10/14/20.
// ThunderGBM cub_wrapper.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/util/cub_wrapper.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_CUB_WRAPPER_H
#define FEDTREE_CUB_WRAPPER_H
#include "thrust/sort.h"
#include "thrust/execution_policy.h"
#include <FedTree/syncarray.h>
#ifdef USE_CUDA
#include "cub/cub.cuh"

template<typename T1, typename T2>
void seg_sort_by_key_cpu(SyncArray<T1> &keys, SyncArray<T2> &values, const SyncArray<int> &ptr) {
    auto keys_data = keys.device_data();
    auto values_data = values.device_data();
    auto offset_data = ptr.host_data();
    LOG(INFO) << ptr;
    for(int i = 0; i < ptr.size() - 2; i++)
    {
        int seg_len = offset_data[i + 1] - offset_data[i];
        auto key_start = keys_data + offset_data[i];
        auto key_end = key_start + seg_len;
        auto value_start = values_data + offset_data[i];
        thrust::sort_by_key(thrust::device, key_start, key_end, value_start, thrust::greater<T1>());
    }
}
#else

template<typename T1, typename T2>
void seg_sort_by_key_cpu(vector<T1> &keys, vector<T2> &values, vector<int> &ptr) {
    auto keys_data = keys.data();
    auto values_data = values.data();
    auto offset_data = ptr.data();
    for(int i = 0; i < ptr.size() - 2; i++)
    {
        int seg_len = offset_data[i + 1] - offset_data[i];
        auto key_start = keys_data + offset_data[i];
        auto key_end = key_start + seg_len;
        auto value_start = values_data + offset_data[i];
        thrust::sort_by_key(thrust::host, key_start, key_end, value_start, thrust::greater<T1>());
    }
}

#endif


#endif //FEDTREE_CUB_WRAPPER_H
