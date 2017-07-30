cmake_minimum_required(VERSION 3.7)
project(kalman2)

set(CMAKE_CXX_STANDARD 14)

include_directories( include src )


file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(kalman2 ${SOURCE_FILES} ${HEADER_FILES})
add_subdirectory(examples)

install(TARGETS kalman2 DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include/kalman)

#first we can indicate the documentation build as an option and set it to ON by default
option(BUILD_DOC "Build documentation" ON)

# check if Doxygen is installed
# needs the graphvis dot command
find_package(Doxygen
             REQUIRED dot)
if (DOXYGEN_FOUND)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/public)
    set(DOXYGEN_RAW_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/../src/Doxyfile.in)
    set(DOXYGEN_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.in)
    # copy ../src/Doxyfile.in to cmake_dir/Doxyfile.in and replace the @variables@
    configure_file(${DOXYGEN_RAW_CONFIG_FILE} ${DOXYGEN_CONFIG_FILE} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
    # cmake 3.9 new function for doxygen
    #set(DOXYGEN_GENERATE_HTML YES)
    #set(DOXYGEN_GENERATE_MAN YES)
    #doxygen_add_docs(doxygen
    #                ${PROJECT_SOURCE_DIR}
    #                COMMENT "Generating API documentation with Doxygen"
    #                )
            else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
