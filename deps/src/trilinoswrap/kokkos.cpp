#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Kokkos_DefaultNode.hpp>

namespace trilinoswrap
{

void register_kokkos(cxx_wrap::Module& mod)
{
  mod.add_type<KokkosClassic::DefaultNode::DefaultNodeType>("DefaultNodeType");
}

} // namespace trilinoswrap
