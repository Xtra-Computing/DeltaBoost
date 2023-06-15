//
// Created by liqinbin on 10/14/20.
// ThunderGBM syncarray.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/syncarray.h
// Under Apache-2.0 License
// Copyright (c) 2020 by jiashuai
//

#ifndef FEDTREE_SYNCARRAY_H
#define FEDTREE_SYNCARRAY_H

#include "FedTree/util/log.h"
#include "syncmem.h"
#include <cstdio>

/**
 * @brief Wrapper of SyncMem with a type
 * @tparam T type of element
 */
template<typename T>
class SyncArray : public el::Loggable {
public:
    /**
     * initialize class that can store given count of elements
     * @param count the given count
     */
    explicit SyncArray(size_t count) : mem(new SyncMem(sizeof(T) * count)), size_(count) {
    }

    SyncArray() : mem(nullptr), size_(0) {}

    ~SyncArray() { delete mem; };

    const T *host_data() const {
        to_host();
        return static_cast<T *>(mem->host_data());
    };

    const T *device_data() const {
        to_device();
        return static_cast<T *>(mem->device_data());
    };

    T *host_data() {
        to_host();
        return static_cast<T *>(mem->host_data());
    };

    T *device_data() {
        to_device();
        return static_cast<T *>(mem->device_data());
    };

    T *device_end() {
        return device_data() + size();
    };

    const T *device_end() const {
        return device_data() + size();
    };

    T *host_end() {
        return host_data() + size();
    };

    const T *host_end() const {
        return host_data() + size();
    }

    void set_host_data(T *host_ptr) {
        mem->set_host_data(host_ptr);
    }

    void set_device_data(T *device_ptr) {
        mem->set_device_data(device_ptr);
    }

    void to_host() const {
        CHECK_GT(size_, 0);
        mem->to_host();
    }

    void to_device() const {
        CHECK_GT(size_, 0);
        mem->to_device();
    }

    /**
     * copy device data. This will call to_device() implicitly.
     * @param source source data pointer (data can be on host or device)
     * @param count the count of elements
     */
    void copy_from(const T *source, size_t count) {

#ifdef USE_CUDA
        thunder::device_mem_copy(mem->device_data(), source, sizeof(T) * count);
#else
        memcpy(mem->host_data(), source, sizeof(T) * count);
#endif
    };

    void copy_from(const SyncArray<T> &source) {

        CHECK_EQ(size(), source.size()) << "destination and source count doesn't match";
#ifdef USE_CUDA
        if (get_owner_id() == source.get_owner_id())
            copy_from(source.device_data(), source.size());
        else
            CUDA_CHECK(cudaMemcpyPeer(mem->device_data(), get_owner_id(), source.device_data(), source.get_owner_id(),
                                      source.mem_size()));
#else
        copy_from(source.host_data(), source.size());
#endif
    };

    /**
     * resize to a new size. This will also clear all data.
     * @param count
     */
    void resize(size_t count) {
        if(mem != nullptr || mem != NULL) {
            delete mem;
        }
        mem = new SyncMem(sizeof(T) * count);
        this->size_ = count;
    };

    /*
     * resize to a new size. This will not clear the origin data.
     * @param count
     */
    void resize_without_delete(size_t count) {
//        delete mem;
        mem = new SyncMem(sizeof(T) * count);
        this->size_ = count;
    };


    size_t mem_size() const {//number of bytes
        return mem->size();
    }

    size_t size() const {//number of values
        return size_;
    }

    SyncMem::HEAD head() const {
        return mem->head();
    }

    void log(el::base::type::ostream_t &ostream) const override {
        int i;
        size_t maxLogPerContainer = 700;
        ostream << "[";
        const T *data = host_data();
        for (i = 0; i < size() - 1 && i < maxLogPerContainer - 1; ++i) {
//    for (i = 0; i < size() - 1; ++i) {
            ostream << data[i] << ",";
        }
        ostream << host_data()[i];
        if (size_ <= maxLogPerContainer) {
            ostream << "]";
        } else {
            ostream << ", ...(" << size_ - maxLogPerContainer << " more)";
        }
    };
#ifdef USE_CUDA
    int get_owner_id() const {
        return mem->get_owner_id();
    }
#endif
    //move constructor
    SyncArray(SyncArray<T> &&rhs) noexcept  : mem(rhs.mem), size_(rhs.size_) {
        rhs.mem = nullptr;
        rhs.size_ = 0;
    }

    //move assign
    SyncArray &operator=(SyncArray<T> &&rhs) noexcept {
        delete mem;
        mem = rhs.mem;
        size_ = rhs.size_;

        rhs.mem = nullptr;
        rhs.size_ = 0;
        return *this;
    }

    SyncArray(const SyncArray<T> &) = delete;

    SyncArray &operator=(const SyncArray<T> &) = delete;

    std::vector<T> to_vec() const {
        std::vector<T> vec_copy(host_data(), host_data() + size());
        return vec_copy;
    }

//    std::vector<T> move_to_vec() const {
//        // move a sync array to a vector, the sync array will be empty after this operation
//        // directly transfer the ownership of the memory to the vector to avoid copy
//        std::vector<T> vec_move;
//        vec_move.resize(size());
//        vec_move.data() = mem->transfer_host_data();
//
//        //empty the sync array
//        size_ = 0;
//        return vec_move;
//    }

    void load_from_vec(const std::vector<T>& vec) {
        this->resize(vec.size());
        this->copy_from(vec.data(), vec.size());
    }

private:
    SyncMem *mem;
    size_t size_;
};

//SyncArray for multiple devices
template<typename T>
class MSyncArray : public std::vector<SyncArray<T>> {
public:
    explicit MSyncArray(size_t n_device) : base_class(n_device) {};

    explicit MSyncArray(size_t n_device, size_t size) : base_class(n_device) {
        for (int i = 0; i < n_device; ++i) {
            this->at(i) = SyncArray<T>(size);
        }
    };

    MSyncArray() : base_class() {};

    //move constructor and assign
    MSyncArray(MSyncArray<T> &&) = default;

    MSyncArray &operator=(MSyncArray<T> &&) = default;

    MSyncArray(const MSyncArray<T> &) = delete;

    MSyncArray &operator=(const MSyncArray<T> &) = delete;

private:
    typedef std::vector<SyncArray<T>> base_class;
};
#endif //FEDTREE_SYNCARRAY_H
