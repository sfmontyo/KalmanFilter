cmake_minimum_required(VERSION 3.7)
project(kalmanExamples)

set(CMAKE_CXX_STANDARD 14)

include_directories( include src )

file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(kalmanExamples ${SOURCE_FILES})

target_link_libraries(kalmanExamples kalman2)

