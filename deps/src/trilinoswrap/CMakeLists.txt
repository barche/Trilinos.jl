project(Trilinos)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_definitions(-DJULIA_ENABLE_THREADING)

find_package(Trilinos)
find_package(JlCxx)
find_package(Eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS}")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_DIR}/../;${Trilinos_LIBRARY_DIRS}")

include_directories(${Trilinos_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_library(trilinoswrap SHARED
  belos.cpp
  benchmark.cpp
  ifpack2.cpp
  module.cpp
  kokkos.cpp
  muelu.cpp
  testing.cpp
  teuchos.cpp
  tpetra.cpp
  thyra.cpp
)
target_link_libraries(trilinoswrap ${Teuchos_LIBRARIES} ${Tpetra_LIBRARIES} ${Stratimikos_LIBRARIES} ${Thyra_LIBRARIES} ${Ifpack2_LIBRARIES} ${MueLu_LIBRARIES} JlCxx::cxxwrap_julia)

install(TARGETS
  trilinoswrap
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
RUNTIME DESTINATION lib)
