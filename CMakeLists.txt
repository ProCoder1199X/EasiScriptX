cmake_minimum_required(VERSION 3.20)
project(ESX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_DOCS "Build documentation" ON)
option(USE_CUDA "Enable CUDA" OFF)

# Dependencies
find_package(Boost 1.85 REQUIRED COMPONENTS spirit)
find_path(EIGEN_INCLUDE_DIR Eigen/Core REQUIRED)
find_package(OpenMP REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(OpenMPI REQUIRED)
find_package(Torch REQUIRED)
find_package(TensorFlow)
if(USE_CUDA)
    enable_language(CUDA)
    find_library(CUBLAS_LIBRARY cublas)
    find_library(NCCL_LIBRARY nccl)
endif()

# Main executable
add_executable(esx 
    src/config.hpp
    src/tensor.hpp
    src/model.hpp
    src/dataset.hpp
    src/distributed.hpp
    src/interpreter.hpp
    src/ast.hpp
    src/parser.hpp
    src/printer.hpp
    src/main.cpp
)
target_include_directories(esx PRIVATE 
    ${Boost_INCLUDE_DIRS} 
    ${EIGEN_INCLUDE_DIR} 
    ${onnxruntime_INCLUDE_DIRS} 
    ${OpenMPI_INCLUDE_DIRS} 
    ${Torch_INCLUDE_DIRS} 
    src
)
target_link_libraries(esx PRIVATE 
    ${Boost_LIBRARIES} 
    OpenMP::OpenMP_CXX 
    onnxruntime::onnxruntime 
    ${OpenMPI_LIBRARIES} 
    ${Torch_LIBRARIES}
)
if(TensorFlow_FOUND)
    target_link_libraries(esx PRIVATE TensorFlow::TensorFlow)
endif()
if(USE_CUDA)
    target_link_libraries(esx PRIVATE ${CUBLAS_LIBRARY} ${NCCL_LIBRARY})
endif()

if(BUILD_TESTS)
    add_executable(test_esx tests/test_esx.cpp)
    target_include_directories(test_esx PRIVATE 
        ${Boost_INCLUDE_DIRS} 
        ${EIGEN_INCLUDE_DIR} 
        ${onnxruntime_INCLUDE_DIRS} 
        ${OpenMPI_INCLUDE_DIRS} 
        ${Torch_INCLUDE_DIRS} 
        src
    )
    target_link_libraries(test_esx PRIVATE 
        ${Boost_LIBRARIES} 
        OpenMP::OpenMP_CXX 
        onnxruntime::onnxruntime 
        ${OpenMPI_LIBRARIES} 
        ${Torch_LIBRARIES}
    )
    if(TensorFlow_FOUND)
        target_link_libraries(test_esx PRIVATE TensorFlow::TensorFlow)
    endif()
    if(USE_CUDA)
        target_link_libraries(test_esx PRIVATE ${CUBLAS_LIBRARY} ${NCCL_LIBRARY})
    endif()
    enable_testing()
    add_test(NAME ESXTests COMMAND test_esx)
endif()

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs src ALL)
endif()