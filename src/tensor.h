#pragma once
#include<vector>
#include<pybind11/pybind11.h>
#include<iostream>
#define torch_int 0
#define torch_float 1
#define torch_double 2

namespace py = pybind11;

class DataStorage
{
private:
    DataStorage(){}
    ~DataStorage(){}
    std::vector<int> tensor_int;
    std::vector<float> tensor_float;
    std::vector<double> tensor_double;
public:
    static DataStorage& getInstance()
    {
        static DataStorage instance;
        return instance;
    }
    DataStorage(const DataStorage&) = delete;
    void operator=(const DataStorage&) = delete;

    //variables
    static int next_id;

    //id->refCount
    std::unordered_map<int, int> refCounts;

    //id->index
    std::unordered_map<int, int> index_map_int;
    std::unordered_map<int, int> index_map_float;
    std::unordered_map<int, int> index_map_double;

    //functions
    int add_tensor(std::vector<int> value);
    int add_tensor(std::vector<float> value);
    int add_tensor(std::vector<double> value);
    void set_element_int(int data_id, int index, int value);
    void set_element_float(int data_id, int index, float value);
    void set_element_double(int data_id, int index, double value);
    void erase_tensor_int(int id, int len);
    void erase_tensor_float(int id, int len);
    void erase_tensor_double(int id, int len);
    void print_elements(int data_id, int type, int len);
    int get_element_int(int data_id, int index);
    float get_element_float(int data_id, int index);
    double get_element_double(int data_id, int index);
    int add(int data_id_1, int data_id_2, int len, int type);
    void delete_tensor(int data_id, int len, int type);
};


template <typename T>
class tensor
{
public:
    tensor(){}
    tensor(const py::list vals, int dtype)
    {
        this->dtype = dtype;
        this->require_grad = false;
        std::vector<T> caches;
        flatten(vals, 0, caches);
        add_tensor(caches);
        this->Data.refCounts[data_id] = 1;
        this->data_len = caches.size();
    }
    ~tensor()
    {
        Data.delete_tensor(data_id, data_len, dtype);
    }
    tensor(tensor<T>&& t)
    {
        this->dtype = t.dtype;
        this->require_grad = t.require_grad;
        this->shape = t.shape;
        this->data_id = t.data_id;
        this->data_len = t.data_len;
        t.data_id = -1;
    }

    //variables
    int dtype;
    bool require_grad;
    std::vector<int> shape;
    int data_id;
    int data_len;

    //functions
    std::shared_ptr<tensor<T>> getitem(int index);
    void setitem(int index, T value);
    void add_tensor(const std::vector<T>& caches);
    void add(tensor<T>& new_tensor, tensor<T>& t2);
    void erase_tensor();
    std::string str();
    std::string str_helper(int depth, int index)
    {
        std::string res = "";
        if(depth == shape.size() - 1)
        {
            res += "[";
            for(int i = 0; i < shape[depth]; i++)
            {
                res += std::to_string(Data.get_element_int(data_id, i + shape[depth] * index));
                if(i != shape[depth] - 1)
                {
                    res += ",";
                }
            }
            res += "]";
        }
        else
        {
            res += "[";
            for(int i = 0; i < shape[depth]; i++)
            {
                res += str_helper(depth + 1, i + shape[depth] * index);
                if(i != shape[depth] - 1)
                {
                    res += ",";
                }
            }
            res += "]";
        }
        return res;
    }
    std::string get_shape()
    {
        std::string res = "[";
        for(int i = 0; i < shape.size(); i++)
        {
            res += std::to_string(shape[i]);
            if(i != shape.size() - 1)
            {
                res += ",";
            }
        }
        res += "]";
        return res;
    }
private:

    //variables
    DataStorage& Data = DataStorage::getInstance();

    //functions
    void flatten(py::list vals, int depth, std::vector<T>& tensor_vals)
    {
        shape.resize(depth + 1);
        shape[depth] = vals.size();
        for(auto val : vals)
        {
            if(py::isinstance<py::list>(val))
            {
                flatten(py::cast<py::list>(val), depth + 1, tensor_vals);
            }
            else
            {
                tensor_vals.push_back(val.cast<T>());
            }
        }
        
    }
};