cmake_minimum_required(VERSION 3.5)
project(supervisory_frame)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)

set(CAFFE_PATH /home/dujiajun/py-faster-rcnn/caffe-fast-rcnn)
set(CUDA_PATH /usr/local/cuda)

include_directories(include)
include_directories(${CAFFE_PATH}/include ${CAFFE_PATH}/build/src)
include_directories(${CUDA_PATH}/include)

link_directories(/usr/local/lib)
link_directories(${CAFFE_PATH}/build/lib)
link_directories(${CUDA_PATH}/lib)
link_directories(${CUDA_PATH}/lib64)

aux_source_directory(src/opencv2 OPENCV_TRACKING_SOURCE_FILES)

set(SOURCE_FILES
		src/real_time_monitor.cpp
        src/tracker/kcf_tracker.cpp
        src/class_independent_tracker.cpp
        src/detector/faster_rcnn_detector.cpp
        src/multi_target_detector.cpp
		src/target.cpp
        ${OPENCV_TRACKING_SOURCE_FILES})

add_executable(real_time_monitor ${SOURCE_FILES})
#add_executable(multi_target_detector src/multi_target_detector.cpp src/target.cpp)

# attention:
# OpenCV_DIR should be set by yourself,
# remember that directory like {opencv-3.1.0/build} is created by you for installing OpenCV
set(OpenCV_DIR /home/dujiajun/opencv-3.2.0/build)
FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(real_time_monitor ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(multi_target_detector ${OpenCV_LIBS})

 find_package(Threads REQUIRED)
 if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET real_time_monitor PROPERTY COMPILE_OPTIONS -pthread)
    set_property(TARGET real_time_monitor PROPERTY INTERFACE_COMPILE_OPTIONS -pthread)
 endif()
 if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(real_time_monitor ${CMAKE_THREAD_LIBS_INIT})
 endif()

#target_link_libraries(multi_target_detector -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcblas -latlas -lcaffe)
target_link_libraries(real_time_monitor -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcblas -latlas -lcaffe)
