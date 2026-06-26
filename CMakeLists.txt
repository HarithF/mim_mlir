cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(mlir)

if(NOT COMMAND add_mim_plugin)
    find_package(mim REQUIRED)
endif()

add_mim_plugin(mlir
    SOURCES
        src/mlir.cpp
        src/normalizers.cpp
    INSTALL
)
