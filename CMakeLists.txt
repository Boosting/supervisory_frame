cmake_minimum_required(VERSION 3.5)
project(supervisory_frame)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)

set(CAFFE_PATH /home/dujiajun/py-faster-rcnn/caffe-fast-rcnn)
set(CUDA_PATH /usr/local/cuda-8.0)
set(HDF5_INCLUDE_PATH /usr/include/hdf5/serial)
include_directories(include)
include_directories(${CAFFE_PATH}/include /home/dujiajun/py-faster-rcnn/caffe-fast-rcnn/.build_release/src/)
include_directories(${CUDA_PATH}/include)
include_directories(${HDF5_INCLUDE_PATH})

set(SOURCE_FILES
    include/class_independent_tracker.hpp
    include/multi_target_detector.hpp
    include/real_time_monitor.hpp
    src/class_independent_tracker.cpp
    src/real_time_monitor.cpp
        src/class_independent_tracker.cpp
        src/target.cpp
)

add_executable(supervisory_frame ${SOURCE_FILES})
add_executable(multi_target_detector src/multi_target_detector.cpp src/target.cpp)

# attention:
# OpenCV_DIR should be set by yourself,
# remember that directory like {opencv-3.1.0/build} is created by you for installing OpenCV
set(OpenCV_DIR /home/dujiajun/opencv-3.2.0/build)
FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(supervisory_frame ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(multi_target_detector ${OpenCV_LIBS})

find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET supervisory_frame PROPERTY COMPILE_OPTIONS -pthread)
    set_property(TARGET supervisory_frame PROPERTY INTERFACE_COMPILE_OPTIONS -pthread)
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(supervisory_frame ${CMAKE_THREAD_LIBS_INIT})
endif()
