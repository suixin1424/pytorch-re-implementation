#include"tensor.h"
#include <algorithm>


//class DataStorage
void DataStorage::erase_tensor_int(int id, int len)
{
    tensor_int.erase(tensor_int.begin() + index_map_int[id], tensor_int.begin() + index_map_int[id] + len);
    //update index_map
    for(auto& [key, value] : index_map_int)
    {
        if(value > index_map_int[id])
        {
            value -= len;
        }
    }
}

void DataStorage::erase_tensor_float(int id, int len)
{
    tensor_float.erase(tensor_float.begin() + index_map_float[id], tensor_float.begin() + index_map_float[id] + len);
    //update index_map
    for(auto& [key, value] : index_map_float)
    {
        if(value > index_map_float[id])
        {
            value -= len;
        }
    }
}

void DataStorage::erase_tensor_double(int id, int len)
{
    tensor_double.erase(tensor_double.begin() + index_map_double[id], tensor_double.begin() + index_map_double[id] + len);
    //update index_map
    for(auto& [key, value] : index_map_double)
    {
        if(value > index_map_double[id])
        {
            value -= len;
        }
    }
}

int DataStorage::add_tensor(std::vector<int> value)
{
    tensor_int.insert(tensor_int.end(), value.begin(), value.end());
    index_map_int[next_id] = tensor_int.size() - value.size();
    next_id++;
    return next_id - 1;
}

int DataStorage::add_tensor(std::vector<float> value)
{
    tensor_float.insert(tensor_float.end(), value.begin(), value.end());
    index_map_float[next_id++] = tensor_float.size() - value.size();
    return next_id - 1;
}

int DataStorage::add_tensor(std::vector<double> value)
{
    tensor_double.insert(tensor_double.end(), value.begin(), value.end());
    index_map_double[next_id++] = tensor_double.size() - value.size();
    return next_id - 1;
}

int DataStorage::get_element_int(int data_id, int index)
{
    return tensor_int[index_map_int[data_id]+index];
}

float DataStorage::get_element_float(int data_id, int index)
{
    return tensor_float[index_map_float[data_id]+index];
}

double DataStorage::get_element_double(int data_id, int index)
{
    return tensor_double[index_map_double[data_id]+index];
}

void DataStorage::set_element_int(int data_id, int index, int value)
{
    //std::cout << data_id << " " << index << " " << value << " " << index_map_int[data_id] << std::endl;
    tensor_int[index_map_int[data_id]+index] = value;
}

void DataStorage::set_element_float(int data_id, int index, float value)
{
    tensor_float[index_map_float[data_id]+index] = value;
}

void DataStorage::set_element_double(int data_id, int index, double value)
{
    tensor_double[index_map_double[data_id]+index] = value;
}

int DataStorage::add(int data_id_1, int data_id_2, int len, int type)
{
    if(type == torch_int)
    {
        int index_1 = this->index_map_int[data_id_1];
        int index_2 = this->index_map_int[data_id_2];
        std::vector<int> result(len);
        std::transform(this->tensor_int.begin()+index_1, this->tensor_int.begin()+index_1+len, this->tensor_int.begin()+index_2, result.begin(), std::plus<int>());
        return this->add_tensor(result);
    }
    else if(type == torch_float)
    {
        int index_1 = this->index_map_float[data_id_1];
        int index_2 = this->index_map_float[data_id_2];
        std::vector<float> result(len);
        std::transform(this->tensor_float.begin()+index_1, this->tensor_float.begin()+index_1+len, this->tensor_float.begin()+index_2, result.begin(), std::plus<float>());
        return this->add_tensor(result);
    }
    else if(type == torch_double)
    {
        int index_1 = this->index_map_double[data_id_1];
        int index_2 = this->index_map_double[data_id_2];
        std::vector<double> result(len);
        std::transform(this->tensor_double.begin()+index_1, this->tensor_double.begin()+index_1+len, this->tensor_double.begin()+index_2, result.begin(), std::plus<double>());
        return this->add_tensor(result);
    }
}

void DataStorage::delete_tensor(int data_id, int len, int type)
{
    //std::cout << "gc, data_id: " << data_id << std::endl;
    if(data_id == -1) 
        return;
    if(type == torch_int)
    {
        auto it = this->refCounts.find(data_id);
        if(it != this->refCounts.end())
        {
            it->second--;
            if(it->second == 0)
            {
                this->erase_tensor_int(data_id, len);
                this->refCounts.erase(data_id);
            }
        }
        this->index_map_int.erase(data_id);
    }
    else if(type == torch_float)
    {
        auto it = this->refCounts.find(data_id);
        if(it != this->refCounts.end())
        {
            it->second--;
            if(it->second == 0)
            {
                this->erase_tensor_float(data_id, len);
                this->refCounts.erase(data_id);
            }
        }
        this->index_map_float.erase(data_id);
    }
    else if(type == torch_double)
    {
        auto it = this->refCounts.find(data_id);
        if(it != this->refCounts.end())
        {
            it->second--;
            if(it->second == 0)
            {
                this->erase_tensor_double(data_id, len);
                this->refCounts.erase(data_id);
            }
        }
        this->index_map_double.erase(data_id);
    }
}

//for debug
void DataStorage::print_elements(int data_id, int type, int len)
{
    if(type == torch_int)
    {
        int index = this->index_map_int[data_id];
        for(int i = index; i < len; i++)
        {
            std::cout << this->tensor_int[i] << ",";
        }
        std::cout << std::endl;
    }
    else if(type == torch_float)
    {
        int index = this->index_map_float[data_id];
        for(int i = index; i < len; i++)
        {
            std::cout << this->tensor_float[i] << ",";
        }
        std::cout << std::endl;
    }
    else if(type == torch_double)
    {
        int index = this->index_map_double[data_id];
        for(int i = index; i < len; i++)
        {
            std::cout << this->tensor_double[i] << ",";
        }
        std::cout << std::endl;
    }
}

int DataStorage::next_id = 0;


//class tensor
template<>
void tensor<int>::add_tensor(const std::vector<int>& caches)
{
    this->data_id = Data.add_tensor(caches);
}

template<>
void tensor<float>::add_tensor(const std::vector<float>& caches)
{
    this->data_id = Data.add_tensor(caches);
}

template<>
void tensor<double>::add_tensor(const std::vector<double>& caches)
{
    this->data_id = Data.add_tensor(caches);
}

template<>
void tensor<int>::erase_tensor()
{
    this->Data.erase_tensor_int(this->data_id, this->data_len);
}

template<>
void tensor<float>::erase_tensor()
{
    this->Data.erase_tensor_float(this->data_id, this->data_len);
}

template<>
void tensor<double>::erase_tensor()
{
    this->Data.erase_tensor_double(this->data_id, this->data_len);
}

template<>
std::shared_ptr<tensor<int>> tensor<int>::getitem(int index)
{
    if(index >= this->shape[0])
    {
        throw std::invalid_argument("index out of range");
    }
    std::shared_ptr<tensor<int>> new_tensor = std::make_shared<tensor<int>>();
    new_tensor->dtype = this->dtype;
    new_tensor->require_grad = this->require_grad;
    new_tensor->shape = this->shape;
    if(new_tensor->shape.size() > 1)
        new_tensor->shape.erase(new_tensor->shape.begin());
    else
        new_tensor->shape[0] = 1;
    
    //find the data_id
    int data_index = this->Data.index_map_int[this->data_id] + index*this->data_len/this->shape[0];
    for(auto& [key, value] : this->Data.index_map_int)
    {
        if(value == data_index)
        {
            new_tensor->data_id = key;
            new_tensor->data_len = this->data_len/this->shape[0];
            auto it = this->Data.refCounts.find(key);
            it->second++;
            return new_tensor;
        }
    }
    //if not found, create a new data_id
    int id = this->Data.next_id++;
    new_tensor->data_id = id;
    new_tensor->data_len = this->data_len/this->shape[0];
    this->Data.index_map_int[id] = data_index;
    this->Data.refCounts[id] = 1;
    return new_tensor;
    //this->Data.print_elements(new_tensor.data_id, torch_int, new_tensor.data_len); 
}

template<>
std::shared_ptr<tensor<float>> tensor<float>::getitem(int index)
{
    if(index >= this->shape[0])
    {
        throw std::invalid_argument("index out of range");
    }
    std::shared_ptr<tensor<float>> new_tensor = std::make_shared<tensor<float>>();
    new_tensor->dtype = this->dtype;
    new_tensor->require_grad = this->require_grad;
    new_tensor->shape = this->shape;
    if(new_tensor->shape.size() > 1)
        new_tensor->shape.erase(new_tensor->shape.begin());
    else
        new_tensor->shape[0] = 1;
    
    //find the data_id
    int data_index = this->Data.index_map_float[this->data_id] + index*this->data_len/this->shape[0];
    for(auto& [key, value] : this->Data.index_map_float)
    {
        if(value == data_index)
        {
            new_tensor->data_id = key;
            new_tensor->data_len = this->data_len/this->shape[0];
            auto it = this->Data.refCounts.find(key);
            it->second++;
            return new_tensor;
        }
    }
    //if not found, create a new data_id
    int id = this->Data.next_id++;
    new_tensor->data_id = id;
    new_tensor->data_len = this->data_len/this->shape[0];
    this->Data.index_map_float[id] = data_index;
    this->Data.refCounts[id] = 1;
    return new_tensor;
    //this->Data.print_elements(new_tensor.data_id, torch_float, new_tensor.data_len); 
}

template<>
std::shared_ptr<tensor<double>> tensor<double>::getitem(int index)
{
    if(index >= this->shape[0])
    {
        throw std::invalid_argument("index out of range");
    }
    std::shared_ptr<tensor<double>> new_tensor = std::make_shared<tensor<double>>();
    new_tensor->dtype = this->dtype;
    new_tensor->require_grad = this->require_grad;
    new_tensor->shape = this->shape;
    if(new_tensor->shape.size() > 1)
        new_tensor->shape.erase(new_tensor->shape.begin());
    else
        new_tensor->shape[0] = 1;
    
    //find the data_id
    int data_index = this->Data.index_map_double[this->data_id] + index*this->data_len/this->shape[0];
    for(auto& [key, value] : this->Data.index_map_double)
    {
        if(value == data_index)
        {
            new_tensor->data_id = key;
            new_tensor->data_len = this->data_len/this->shape[0];
            auto it = this->Data.refCounts.find(key);
            it->second++;
            return new_tensor;
        }
    }
    //if not found, create a new data_id
    int id = this->Data.next_id++;
    new_tensor->data_id = id;
    new_tensor->data_len = this->data_len/this->shape[0];
    this->Data.index_map_double[id] = data_index;
    this->Data.refCounts[id] = 1;
    return new_tensor;
    //this->Data.print_elements(new_tensor.data_id, torch_double, new_tensor.data_len); 
}

template<>
std::string tensor<int>::str()
{
    std::string res = "tensor_int(";
    res += this->str_helper(0, 0);
    res+=")";
    return res;
}

template<>
std::string tensor<float>::str()
{
    std::string res = "tensor_float(";
    res += this->str_helper(0, 0);
    res+=")";
    return res;
}

template<>
std::string tensor<double>::str()
{
    std::string res = "tensor_double(";
    res += this->str_helper(0, 0);
    res+=")";
    return res;
}

template<>
void tensor<int>::setitem(int index, int value)
{
    if(this->shape.size() != 1)
    {
        throw std::invalid_argument("only 1-d tensor can be set");
    }
    if(index >= this->shape[0])
    {
        throw std::invalid_argument("index out of range");
    }
    this->Data.set_element_int(this->data_id, index, value);
}

template<>
void tensor<float>::setitem(int index, float value)
{
    if(this->shape.size() != 1)
    {
        throw std::invalid_argument("only 1-d tensor can be set");
    }
    if(index >= this->shape[0])
    {
        throw std::invalid_argument("index out of range");
    }
    this->Data.set_element_float(this->data_id, index, value);
}

template<>
void tensor<double>::setitem(int index, double value)
{
    if(this->shape.size() != 1)
    {
        throw std::invalid_argument("only 1-d tensor can be set");
    }
    if(index >= this->shape[0])
    {
        throw std::invalid_argument("index out of range");
    }
    this->Data.set_element_double(this->data_id, index, value);
}

template<>
void tensor<int>::add(tensor<int>& new_tensor, tensor<int>& t2)
{
    if(this->shape != t2.shape)
    {
        throw std::invalid_argument("shape not match");
    }
    if(this->dtype != t2.dtype)
    {
        throw std::invalid_argument("dtype not match");
    }
    int data_id = Data.add(this->data_id, t2.data_id, this->data_len, torch_int);
    new_tensor.dtype = this->dtype;
    new_tensor.require_grad = this->require_grad;
    new_tensor.shape = this->shape;
    new_tensor.data_id = data_id;
    new_tensor.data_len = this->data_len;
}

template<>
void tensor<float>::add(tensor<float>& new_tensor, tensor<float>& t2)
{
    if(this->shape != t2.shape)
    {
        throw std::invalid_argument("shape not match");
    }
    if(this->dtype != t2.dtype)
    {
        throw std::invalid_argument("dtype not match");
    }
    int data_id = Data.add(this->data_id, t2.data_id, this->data_len, torch_float);
    new_tensor.dtype = this->dtype;
    new_tensor.require_grad = this->require_grad;
    new_tensor.shape = this->shape;
    new_tensor.data_id = data_id;
    new_tensor.data_len = this->data_len;
}

template<>
void tensor<double>::add(tensor<double>& new_tensor, tensor<double>& t2)
{
    if(this->shape != t2.shape)
    {
        throw std::invalid_argument("shape not match");
    }
    if(this->dtype != t2.dtype)
    {
        throw std::invalid_argument("dtype not match");
    }
    int data_id = Data.add(this->data_id, t2.data_id, this->data_len, torch_double);
    new_tensor.dtype = this->dtype;
    new_tensor.require_grad = this->require_grad;
    new_tensor.shape = this->shape;
    new_tensor.data_id = data_id;
    new_tensor.data_len = this->data_len;
}
