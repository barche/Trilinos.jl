project(Trilinos)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Trilinos)
find_package(Eigen3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${Trilinos_CXX_COMPILER_FLAGS}")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${Trilinos_LIBRARY_DIRS}")

include_directories(${Trilinos_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(powermethod powermethod.cpp)
target_link_libraries(powermethod ${Teuchos_LIBRARIES} ${Tpetra_LIBRARIES})

add_executable(laplace2d laplace2d.cpp)
target_link_libraries(laplace2d ${Teuchos_LIBRARIES} ${Tpetra_LIBRARIES} ${Thyra_LIBRARIES} ${Belos_LIBRARIES})
