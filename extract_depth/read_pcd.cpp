/* CPP program that reads a point cloud data file (from cornell grasp dataset)
    and returns a numpy array back to the calling python program */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
#include <string>
#include <fstream>
#include <regex>

using namespace std;

static PyObject* pcdreader_read(PyObject *self, PyObject *args)
{
    /*Numpy extension that will read a point cloud data file
    and return a numpy tensor. The point cloud files contain
    data from the cornell grasp dataset.*/

    const char *path;

    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;

    string inp;
    regex space("\\s+|\\n+");
    ifstream ifs(path, ifstream::in);
    regex_token_iterator<string::iterator> end;
    npy_intp dims[3] {480, 640, 3};

    //this leads to a memory leak if the file does not exist, fix later
    float *arr = new float [480*640*3];

    if(!ifs) {
        cout << "Error " << path << " not found\n";
        Py_INCREF(Py_None);
        return Py_None;
    }

    else {
        //first skip the header
        for(int i{0};i<10;++i){
            getline(ifs, inp);
            if(ifs.eof())
                break;
        }

        //read the point cloud file line by line and create a tensor
        while(ifs) {
            getline(ifs, inp);
            if(ifs.eof())
                break;
            if(inp.length() < 1)
                continue;
            regex_token_iterator<string::iterator> it(inp.begin(), inp.end(), space, -1);

            float x, y, z;
            int ind, row, column;
            x = stof(*it);
            ++it;
            y = stof(*it);
            ++it;
            z = stof(*it);
            ++it;
            ++it;
            ind = stoi(*it);
            row = ind / 640;
            column = ind % 640;
            ind = 3*(row*640 + column);
            arr[ind] = x;
            arr[ind+1] = y;
            arr[ind+2] = z;
        }
        ifs.close();

        PyObject *narr = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, reinterpret_cast<void*>(arr));
        return narr;
    }
}

static PyMethodDef PCDMethods[] = {
    {"read",  pcdreader_read, METH_VARARGS,
    "read point cloud file into a numpy array"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef pcdmodule = {
    PyModuleDef_HEAD_INIT,
    "pcdreader",
    NULL,
    -1,
    PCDMethods
};

PyMODINIT_FUNC PyInit_pcdreader(void)
{
    import_array();
    return PyModule_Create(&pcdmodule);
}

int main()
{
    return 0;
}
