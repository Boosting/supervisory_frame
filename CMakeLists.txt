cmake_minimum_required(VERSION 3.5)
project(supervisory_frame)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)

set(CAFFE_PATH /home/dujiajun/py-faster-rcnn/caffe-fast-rcnn)
set(HDF5_PATH /usr/include/hdf5/serial/)
set(CUDA_PATH /usr/local/cuda)

include_directories(include)
include_directories(${HDF5_PATH})
include_directories(${CAFFE_PATH}/include ${CAFFE_PATH}/build/src)
include_directories(${CUDA_PATH}/include)

link_directories(/usr/local/lib)
link_directories(${CAFFE_PATH}/build/lib)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial/)
link_directories(${CUDA_PATH}/lib)
link_directories(${CUDA_PATH}/lib64)

aux_source_directory(src/opencv2 OPENCV_TRACKING_SOURCE_FILES)

add_executable(demo
        src/demo.cpp
        src/real_time_monitor.cpp
        src/displayer.cpp
        src/fusion/detector_only_fusion.cpp
        src/detector/faster_rcnn_detector.cpp
        src/detector/caffe_detector.cpp
        src/target.cpp
        src/utils/opencv_util.cpp)

# attention:
# OpenCV_DIR should be set by yourself,
# remember that directory like {opencv-3.2.0/build} is created by you for installing OpenCV
set(OpenCV_DIR /home/dujiajun/opencv-3.2.0/build)
FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(demo ${OpenCV_LIBS})

find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET demo PROPERTY COMPILE_OPTIONS -pthread)
    set_property(TARGET demo PROPERTY INTERFACE_COMPILE_OPTIONS -pthread)
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(demo ${CMAKE_THREAD_LIBS_INIT})
endif()

target_link_libraries(demo -lopencv_core -lopencv_highgui -lopencv_imgproc)
target_link_libraries(demo -lboost_system -lboost_thread -lstdc++)
target_link_libraries(demo -lcudart -lcublas -lcurand)
target_link_libraries(demo -lcaffe -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lhdf5_hl -lhdf5 -lm -lcblas -latlas)
