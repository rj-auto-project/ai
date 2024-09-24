cmake_minimum_required(VERSION 3.15)
project(ai_cam_c LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# set(Torch_DIR "/home/annone/libtorch/share/cmake/Torch/")

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(example-app /home/annone/example-app/example-app.cpp)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")
# set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.


# Specify the path to LibTorch
set(CMAKE_PREFIX_PATH "/home/annone/libtorch")  # Correct path to LibTorch

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Torch
set(Torch_DIR "/home/annone/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED)

# Find Redis (hiredis)
find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIB hiredis)

if(NOT HIREDIS_INCLUDE_DIR OR NOT HIREDIS_LIB)
    message(FATAL_ERROR "Could not find hiredis")
endif()

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories for OpenCV, Torch, Hiredis, and CUDA
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${HIREDIS_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Add executable
add_executable(ai_cam_c main.cpp)

# Link OpenCV, Torch, Hiredis, and CUDA libraries
target_link_libraries(ai_cam_c
    ${OpenCV_LIBS}
    "${TORCH_LIBRARIES}"
    ${HIREDIS_LIB}
    ${CUDA_LIBRARIES}
)

set_property(TARGET ai_cam_c PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ai_cam_c
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:ai_cam_c>)
endif (MSVC)

# Set compiler flags required by Torch and CUDA
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${CUDA_CXX_FLAGS}")
