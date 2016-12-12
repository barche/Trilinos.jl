project(Trilinos)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
add_definitions(-DJULIA_ENABLE_THREADING)

find_package(Trilinos)
find_package(CxxWrap)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CxxWrap_DIR}/../;${Trilinos_LIBRARY_DIRS}")

add_library(trilinoswrap SHARED
  module.cpp
  teuchos.cpp)
target_link_libraries(trilinoswrap ${Teuchos_LIBRARIES} CxxWrap::cxx_wrap)

install(TARGETS
  trilinoswrap
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
RUNTIME DESTINATION lib)
