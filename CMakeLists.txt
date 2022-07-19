cmake_minimum_required(VERSION 3.20)
project(pipe_transmission VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)
set(OPENCV_COMPONENTS
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        opencv_objdetect
        opencv_video)

add_library(TransLib utils/ImageSender.cpp utils/RGBDReceiver.cpp utils/FileRGBDReceiver.cpp utils/FileRGBDReceiver.h utils/RGBDData.cpp utils/RGBDData.h)
set_target_properties(TransLib PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/utils/RGBDReceiver.h)

add_executable(receive
        TestRGBDReceive.cpp)
target_link_libraries(receive
        ${OPENCV_COMPONENTS}
        TransLib
        pthread
        )

add_executable(send
        TestImageSend.cpp
        utils/RGBDReceiver.cpp utils/ImageSender.cpp utils/ImageSender.h)
target_link_libraries(send
        ${OPENCV_COMPONENTS}
        TransLib
        pthread
        )
