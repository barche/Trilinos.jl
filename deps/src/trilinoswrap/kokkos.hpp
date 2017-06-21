#ifndef TRILINOS_JL_KOKKOS_HPP
#define TRILINOS_JL_KOKKOS_HPP

#include "jlcxx/jlcxx.hpp"

#include <Kokkos_DefaultNode.hpp>
#include <TpetraCore_config.h>

namespace trilinoswrap
{

typedef jlcxx::ParameterList<int> local_ordinals_t;
typedef jlcxx::ParameterList<int, int64_t> global_ordinals_t;
typedef jlcxx::ParameterList<double> scalars_t;
typedef jlcxx::ParameterList<double**> arrays_t;

#if defined(HAVE_TPETRA_INST_SERIAL) && !defined(HAVE_TPETRA_INST_OPENMP)
  typedef jlcxx::ParameterList<Kokkos::Compat::KokkosSerialWrapperNode> kokkos_nodes_t;
  typedef jlcxx::ParameterList<Kokkos::Serial> kokkos_devices_t;
#elif !defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_OPENMP)
  typedef jlcxx::ParameterList<Kokkos::Compat::KokkosOpenMPWrapperNode> kokkos_nodes_t;
  typedef jlcxx::ParameterList<Kokkos::OpenMP> kokkos_devices_t;
#elif defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_OPENMP)
  typedef jlcxx::ParameterList<Kokkos::Compat::KokkosSerialWrapperNode, Kokkos::Compat::KokkosOpenMPWrapperNode> kokkos_nodes_t;
  typedef jlcxx::ParameterList<Kokkos::Serial, Kokkos::OpenMP> kokkos_devices_t;
#endif

} // namespace trilinoswrap

#endif
