#include <cxx_wrap.hpp>

namespace trilinoswrap
{

extern void register_kokkos(cxx_wrap::Module& mod);
extern void register_teuchos(cxx_wrap::Module& mod);
extern void register_tpetra(cxx_wrap::Module& mod);
extern void register_thyra(cxx_wrap::Module& mod);

} // namespace trilinoswrap


JULIA_CPP_MODULE_BEGIN(registry)
  using namespace cxx_wrap;

  trilinoswrap::register_kokkos(registry.create_module("Kokkos"));
  trilinoswrap::register_teuchos(registry.create_module("Teuchos"));
  trilinoswrap::register_tpetra(registry.create_module("Tpetra"));
  trilinoswrap::register_thyra(registry.create_module("Thyra"));
JULIA_CPP_MODULE_END
