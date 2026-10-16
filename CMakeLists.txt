cmake_minimum_required(VERSION 4.2)
project(SVM)
set(CMAKE_CXX_STANDARD     20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenMP REQUIRED)
add_library(SVM_lib STATIC core/src/SVC.cpp core/src/Kernel.cpp core/src/SVR.cpp)
target_include_directories(SVM_lib PUBLIC core/include)
target_link_libraries(SVM_lib PUBLIC 
    OpenMP::OpenMP_CXX
)
if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(cpp_tests)
endif()


option(BUILD_PYTHON "Build Python bindings via Cython" ON)
if(BUILD_PYTHON)

# 3. Python Bindings (Built when scikit-build-core invokes CMake)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
# Custom command to generate SVM.cpp from SVM.pyx
set(CYTHON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/SVM.cpp")
add_custom_command(
    OUTPUT ${CYTHON_OUTPUT}
    DEPENDS python/SVM.pyx python/SVM.pxd
    COMMAND ${Python_EXECUTABLE} -m cython --cplus --module-name svm_cpp ${CMAKE_CURRENT_SOURCE_DIR}/python/SVM.pyx -o ${CYTHON_OUTPUT}
    COMMENT "Cythonizing SVM.pyx to C++"
)

# Create the Python extension module
Python_add_library(SVM_py MODULE ${CYTHON_OUTPUT} WITH_SOABI)
target_link_libraries(SVM_py PRIVATE SVM_lib Python::NumPy)
target_include_directories(SVM_py PRIVATE core/include)

# Python modules typically shouldn't have a 'lib' prefix on their filename (e.g., '_SVM.so' instead of 'lib_SVM.so')
set_target_properties(SVM_py PROPERTIES 
    PREFIX ""
    OUTPUT_NAME "svm_cpp")

# Tell scikit-build-core to include the compiled extension module in the package
install(TARGETS SVM_py DESTINATION .)
endif()