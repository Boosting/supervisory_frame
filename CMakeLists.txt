cmake_minimum_required(VERSION 3.5)
project(supervisory_frame)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)

set(CAFFE_PATH /home/dujiajun/fast-rcnn/caffe-fast-rcnn)
set(HDF5_PATH /usr/include/hdf5/serial/)
set(CUDA_PATH /usr/local/cuda)
set(DARKNET_PATH /home/dujiajun/darknet)


include_directories(include)
include_directories(${HDF5_PATH})
include_directories(${CAFFE_PATH}/include ${CAFFE_PATH}/build/src)
include_directories(${CUDA_PATH}/include)
include_directories(${DARKNET_PATH}/src)

link_directories(/usr/local/lib)
link_directories(${CAFFE_PATH}/build/lib)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial/)
link_directories(${CUDA_PATH}/lib)
link_directories(${CUDA_PATH}/lib64)
link_directories(${DARKNET_PATH})

aux_source_directory(src/opencv2 OPENCV_TRACKING_SOURCE_FILES)

set(SOURCE_FILES
        src/test_fast_rcnn_detector.cpp
        src/real_time_monitor.cpp
        src/displayer.cpp
        src/fusion/detector_only_fusion.cpp
        src/detector/fast_rcnn_detector.cpp
        src/detector/caffe_detector.cpp
        src/motion_detector/background_substraction_motion_detector.cpp
        src/target.cpp
        src/utils/opencv_util.cpp)

add_executable(test_fast_rcnn_detector ${SOURCE_FILES})

# attention:
# OpenCV_DIR should be set by yourself,
# remember that directory like {opencv-3.2.0/build} is created by you for installing OpenCV
set(OpenCV_DIR /home/dujiajun/opencv-3.2.0/build)
FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(test_fast_rcnn_detector ${OpenCV_LIBS})

find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET test_fast_rcnn_detector PROPERTY COMPILE_OPTIONS -pthread)
    set_property(TARGET test_fast_rcnn_detector PROPERTY INTERFACE_COMPILE_OPTIONS -pthread)
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(test_fast_rcnn_detector ${CMAKE_THREAD_LIBS_INIT})
endif()

target_link_libraries(test_fast_rcnn_detector -ldarknet)
target_link_libraries(test_fast_rcnn_detector -lopencv_core -lopencv_highgui -lopencv_imgproc)
target_link_libraries(test_fast_rcnn_detector -lboost_system -lboost_thread -lstdc++)
target_link_libraries(test_fast_rcnn_detector -lcudart -lcublas -lcurand)
target_link_libraries(test_fast_rcnn_detector -lcaffe -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lhdf5_hl -lhdf5 -lm -lcblas -latlas)
