#include </usr/include/python3.7/Python.h>

float* extract_data(char* data_receive)
{
    static float temp[8];
    int index = 0;
    
    char * token = strtok(data_receive, ",");
    while( token != NULL ) {
        temp[index] = strtof(token, NULL);
        token = strtok(NULL, ",");
        index++;
    }
    
    return temp;
}

int main(int argc, char *argv[]) {
    /* setup */
    char data_receive[20] = "26,23,12,22"; 

    Py_Initialize();
    /* adding current path to sys.path */
    // sys.path 
    PyObject* sysPath = PySys_GetObject("path");  
    // sys.path.insert(0, ".")
    PyList_Insert(sysPath, 0, PyUnicode_FromString("."));  
    // module_string =  "add"
    PyObject* module_string = PyUnicode_FromString((char*)"tflite"); 
    // py_module = __import__(module_string)
    PyObject* py_module = PyImport_Import(module_string);                             
    /* calling function */
    // args =  (1, 2)
    if (py_module == NULL) {
    printf("ERROR importing module\n");
    exit(-1);
    }
    else {
	    printf("Import module success\n");
    }
    
    float* data_split = extract_data(data_receive);
    
    if (data_split == NULL) {
    printf("ERROR split data\n");
    exit(-1);
    }
    
    printf("Data receive: %.1f\n", data_split[0]);
    //PyObject* args = Py_BuildValue("ii",(int)data_split[0],26);
    PyObject* args = Py_BuildValue("iiiiiiii", 26, 27,28,29,30,31,32,33);
    
    // py_function = getattr("add", "add_function")
    PyObject* py_function = PyObject_GetAttrString(py_module,(char*)"predict_list_params");   
    
    if (py_function == NULL) {
    printf("ERROR get function\n");
    exit(-1);
    }
    // py_result  = py_function(*args)
    PyObject* py_result = PyObject_CallObject(py_function, args); 
    
    if (py_result == NULL) {
    printf("ERROR call function\n");
    exit(-1);
    }
    /* converting python result to int in c*/
    //char* result  =  (char*) PyFloat_AsDouble(py_result);
    //const char* result  = PyUnicode_AsUTF8(py_result);
    printf("Result predict: %s\n", PyUnicode_AsUTF8(py_result));
    /* teatdown */
    Py_FinalizeEx();
    return 0;
}
