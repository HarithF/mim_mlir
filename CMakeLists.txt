cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(mlir)

if(NOT COMMAND add_mim_plugin)
  find_package(mim REQUIRED)
endif()

add_mim_plugin(mlir
    SOURCES
        src/mlir.cpp
        src/normalizers.cpp
        src/mlir/lam_classifier.cpp
        src/mlir/mlir_emitter_core.cpp
        src/mlir/mlir_emitter_control_flow.cpp
        src/mlir/mlir_emitter_linalg.cpp
        src/mlir/printer.cpp
        src/mlir/region_tree.cpp
        src/mlir/type_converter.cpp
    INSTALL
)
