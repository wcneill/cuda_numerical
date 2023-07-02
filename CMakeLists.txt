cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda-numeric
  VERSION 1.0
  DESCRIPTION "Cuda accelerated numerical methods"
  LANGUAGES CXX CUDA)

add_executable(test test.cpp integrate.cpp cu_integrate cu_ops.cu)
target_include_directories(test PRIVATE "${CMAKE_SOURCE_DIR}") 