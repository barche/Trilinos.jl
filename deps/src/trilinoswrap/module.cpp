#include <cxx_wrap.hpp>

namespace trilinoswrap
{

extern void register_kokkos(cxx_wrap::Module& mod);
extern void register_teuchos(cxx_wrap::Module& mod);
extern void register_tpetra(cxx_wrap::Module& mod);
extern void register_thyra(cxx_wrap::Module& mod);
extern void register_benchmark(cxx_wrap::Module& mod);

} // namespace trilinoswrap


JULIA_CPP_MODULE_BEGIN(registry)
  using namespace cxx_wrap;

  if(cxx_wrap::symbol_name(jl_current_module->name) == "Kokkos")
  {
    trilinoswrap::register_kokkos(registry.create_module("Kokkos"));
  }
  if(cxx_wrap::symbol_name(jl_current_module->name) == "Teuchos")
  {
    trilinoswrap::register_teuchos(registry.create_module("Teuchos"));
  }
  if(cxx_wrap::symbol_name(jl_current_module->name) == "Tpetra")
  {
    trilinoswrap::register_tpetra(registry.create_module("Tpetra"));
  }
  if(cxx_wrap::symbol_name(jl_current_module->name) == "Thyra")
  {
    trilinoswrap::register_thyra(registry.create_module("Thyra"));
  }
  if(cxx_wrap::symbol_name(jl_current_module->name) == "Benchmark")
  {
    trilinoswrap::register_benchmark(registry.create_module("Benchmark"));
  }
JULIA_CPP_MODULE_END
