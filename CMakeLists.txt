#------------------------------------------------------------------------------
#  SPADES - Simple plug-in architecture using DLLs
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)

project(SPADES)
set(SPADES_VERSION_MAJOR 0)
set(SPADES_VERSION_MAJOR 1)
set(SPADES_VERSION_PATCH 0)


set(SPADES_APP_NAME "SPADES : Simple  plug-ins")
find_package(OpenCL REQUIRED)

include (./cmake/FindOpenCL.cmake)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/UspPlugin/")
include_directories("${PROJECT_SOURCE_DIR}/plugin_b/")


include_directories(${OPENCL_INCLUDE_DIRS})


set (LIBRARY_INSTALL_DIR "lib")
set (INCLUDE_INSTALL_DIR "include")
set (PLUGIN_INSTALL_DIR  "bin/plugins")

add_subdirectory(Plugin_A)
add_subdirectory(Plugin_B)
add_subdirectory(TheApplication)