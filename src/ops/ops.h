#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include "../tensor.h"

template <typename T>
std::shared_ptr<tensor<T>> add_tensor(tensor<T> &t1, tensor<T> &t2)
{
    if(t1.shape != t2.shape)
    {
        throw std::invalid_argument("shape not match");
    }
    if(t1.dtype != t2.dtype)
    {
        throw std::invalid_argument("dtype not match");
    }
    DataStorage& Data = DataStorage::getInstance();
    std::shared_ptr<tensor<T>> new_tensor = std::make_shared<tensor<T>>();
    int data_id = Data.add(t1.data_id, t2.data_id, t1.data_len, t1.dtype);
    new_tensor->dtype = t1.dtype;
    new_tensor->require_grad = t1.require_grad;
    new_tensor->shape = t1.shape;
    new_tensor->data_id = data_id;
    new_tensor->data_len = t1.data_len;
    return new_tensor;
}