#ifndef TRILINOS_JL_KOKKOS_HPP
#define TRILINOS_JL_KOKKOS_HPP

#include <cxx_wrap.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <TpetraCore_config.h>

namespace trilinoswrap
{

typedef cxx_wrap::ParameterList<int> local_ordinals_t;
typedef cxx_wrap::ParameterList<int, int64_t> global_ordinals_t;
typedef cxx_wrap::ParameterList<double> scalars_t;

#if defined(HAVE_TPETRA_INST_SERIAL) && !defined(HAVE_TPETRA_INST_OPENMP)
  typedef cxx_wrap::ParameterList<Kokkos::Compat::KokkosSerialWrapperNode> kokkos_nodes_t;
#elif !defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_OPENMP)
  typedef cxx_wrap::ParameterList<Kokkos::Compat::KokkosOpenMPWrapperNode> kokkos_nodes_t;
#elif defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_OPENMP)
  typedef cxx_wrap::ParameterList<Kokkos::Compat::KokkosSerialWrapperNode, Kokkos::Compat::KokkosOpenMPWrapperNode> kokkos_nodes_t;
#endif

} // namespace trilinoswrap

#endif
