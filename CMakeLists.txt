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
enable_testing()
add_subdirectory(tests)
