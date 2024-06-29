#include <pybind11/pybind11.h>
#include "tensor.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)



PYBIND11_MODULE(ftorch, m) {
    m.doc() = R"pbdoc(
        ftorch
    )pbdoc";
    
    //constants
    m.attr("int") = torch_int;
    m.attr("float") = torch_float;
    m.attr("double") = torch_double;


    //class
    py::class_<tensor<int>, std::shared_ptr<tensor<int>>>(m, "tensor_int")
        .def(py::init<const py::list&, int>())
        .def("__getitem__", [](tensor<int>& t, int index){
            auto value = t.getitem(index);
            return value;
        })
        .def("__setitem__", [](tensor<int>& t, int index, int value){
            t.setitem(index, value);
        })
        .def("__str__", &tensor<int>::str)
        .def("__add__", [](tensor<int>& t1, tensor<int>& t2){
            tensor<int> new_tensor;
            t1.add(new_tensor, t2);
            return new_tensor;
        })
        .def_property_readonly("shape", &tensor<int>::get_shape);
    
    py::class_<tensor<float>, std::shared_ptr<tensor<float>>>(m, "tensor_float")
        .def(py::init<const py::list&, int>())
        .def("__str__", &tensor<float>::str)
        .def("__getitem__", [](tensor<float>& t, int index){
            auto value = t.getitem(index);
            return value;
        })
        .def("__add__", [](tensor<float>& t1, tensor<float>& t2){
            tensor<float> new_tensor;
            t1.add(new_tensor, t2);
            return new_tensor;
        })
        .def_property_readonly("shape", &tensor<float>::get_shape);
        
    py::class_<tensor<double>, std::shared_ptr<tensor<double>>>(m, "tensor_double")
        .def(py::init<const py::list&, int>())
        .def("__str__", &tensor<double>::str)
        .def("__getitem__", [](tensor<double>& t, int index){
            auto value = t.getitem(index);
            return value;
        })
        .def("__add__", [](tensor<double>& t1, tensor<double>& t2){
            tensor<double> new_tensor;
            t1.add(new_tensor, t2);
            return new_tensor;
        })
        .def_property_readonly("shape", &tensor<double>::get_shape);
       
    
    //function
    m.def("tensor", [](const py::list& values, int dtype=torch_int) {
        if (dtype == torch_int)
        {
            return py::cast(tensor<int>(values, dtype));
        }
        else if (dtype == torch_float)
        {
            return py::cast(tensor<float>(values, dtype));
        }
        else if (dtype == torch_double)
        {
            return py::cast(tensor<double>(values, dtype));
        }
        else
        {
            throw std::invalid_argument("dtype must be one of torch_int, torch_float, torch_double");
        }
    }, py::arg("values"), py::arg("dtype")=torch_int);
    

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
