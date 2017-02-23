#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Kokkos_DefaultNode.hpp>
#include <TpetraCore_config.h>

namespace trilinoswrap
{

void register_kokkos(cxx_wrap::Module& mod)
{
#ifdef HAVE_TPETRA_INST_SERIAL
  mod.add_type<Kokkos::Compat::KokkosSerialWrapperNode>("KokkosSerialWrapperNode");
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
  mod.add_type<Kokkos::Compat::KokkosOpenMPWrapperNode>("KokkosOpenMPWrapperNode");
#endif

  mod.method("default_node_type", [] () { return cxx_wrap::julia_type<KokkosClassic::DefaultNode::DefaultNodeType>(); });
}

} // namespace trilinoswrap
