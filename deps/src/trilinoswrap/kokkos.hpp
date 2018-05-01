#ifndef TRILINOS_JL_KOKKOS_HPP
#define TRILINOS_JL_KOKKOS_HPP

#include <Kokkos_DefaultNode.hpp>
#include <TpetraCore_config.h>

#include "jlcxx/jlcxx.hpp"

namespace trilinoswrap
{

typedef jlcxx::ParameterList<int> local_ordinals_t;
typedef jlcxx::ParameterList<int, int64_t> global_ordinals_t;
typedef jlcxx::ParameterList<double> scalars_t;
typedef jlcxx::ParameterList<double**, int*, const unsigned long*> arrays_t;

#if defined(HAVE_TPETRA_INST_SERIAL) && !defined(HAVE_TPETRA_INST_OPENMP)
  typedef jlcxx::ParameterList<Kokkos::Compat::KokkosSerialWrapperNode> kokkos_nodes_t;
  typedef jlcxx::ParameterList<Kokkos::Serial, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>> kokkos_devices_t;
#elif !defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_OPENMP)
  typedef jlcxx::ParameterList<Kokkos::Compat::KokkosOpenMPWrapperNode> kokkos_nodes_t;
  typedef jlcxx::ParameterList<Kokkos::OpenMP, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>> kokkos_devices_t;
#elif defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_OPENMP)
  typedef jlcxx::ParameterList<Kokkos::Compat::KokkosSerialWrapperNode, Kokkos::Compat::KokkosOpenMPWrapperNode> kokkos_nodes_t;
  typedef jlcxx::ParameterList<Kokkos::Serial, Kokkos::OpenMP, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>> kokkos_devices_t;
#endif

} // namespace trilinoswrap

#endif
