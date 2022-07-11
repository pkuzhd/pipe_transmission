cmake_minimum_required(VERSION 3.20)
project(pipe_transmission)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)
set(OPENCV_COMPONENTS
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        opencv_objdetect
        opencv_video)

add_executable(receive
        TestReceive.cpp
        receiver/RGBDReceiver.cpp)
target_link_libraries(receive
        ${OPENCV_COMPONENTS}
        )

add_executable(send
        TestSend.cpp
        receiver/RGBDReceiver.cpp)
target_link_libraries(send
        ${OPENCV_COMPONENTS}
        )