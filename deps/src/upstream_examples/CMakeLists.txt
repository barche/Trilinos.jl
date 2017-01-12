project(Trilinos)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Trilinos)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${Trilinos_LIBRARY_DIRS}")

include_directories(${Trilinos_INCLUDE_DIRS})

add_executable(powermethod powermethod.cpp)
target_link_libraries(powermethod ${Teuchos_LIBRARIES} ${Tpetra_LIBRARIES})
