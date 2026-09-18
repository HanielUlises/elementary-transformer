cmake_minimum_required(VERSION 3.18)
project(elementary_transformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ET_BUILD_PYTHON "Build the pybind11 extension module" OFF)
option(ET_BUILD_TESTS "Build the standalone C++ tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(et_core STATIC cpp/src/refinement.cpp)
target_include_directories(et_core PUBLIC cpp/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(et_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ET_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core cpp/src/bindings.cpp)
  target_link_libraries(_core PRIVATE et_core)
  install(TARGETS _core LIBRARY DESTINATION elementary_transformer)
endif()

if(ET_BUILD_TESTS)
  enable_testing()
  add_executable(et_core_tests cpp/tests/test_refinement.cpp)
  target_link_libraries(et_core_tests PRIVATE et_core)
  add_test(NAME et_core_tests COMMAND et_core_tests)
endif()
