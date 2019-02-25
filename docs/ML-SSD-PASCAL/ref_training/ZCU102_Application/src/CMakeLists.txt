cmake_minimum_required(VERSION 3.8)
project(src)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.cc
        neon_math.hpp
        prior_boxes.cpp
        prior_boxes.hpp
        ssd_detector.cpp
        ssd_detector.hpp
        time_helper.hpp)

add_executable(src ${SOURCE_FILES})