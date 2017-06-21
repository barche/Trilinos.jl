#include "jlcxx/jlcxx.hpp"

namespace trilinoswrap
{

extern void register_belos(jlcxx::Module& mod);
extern void register_ifpack2(jlcxx::Module& mod);
extern void register_kokkos(jlcxx::Module& mod);
extern void register_teuchos(jlcxx::Module& mod);
extern void register_tpetra(jlcxx::Module& mod);
extern void register_thyra(jlcxx::Module& mod);

extern void register_benchmark(jlcxx::Module& mod);
extern void register_testing(jlcxx::Module& mod);

} // namespace trilinoswrap


JULIA_CPP_MODULE_BEGIN(registry)
  using namespace jlcxx;

  if(jlcxx::symbol_name(jl_current_module->name) == "Belos")
  {
    trilinoswrap::register_belos(registry.create_module("Belos"));
  }
  if(jlcxx::symbol_name(jl_current_module->name) == "Ifpack2")
  {
    trilinoswrap::register_ifpack2(registry.create_module("Ifpack2"));
  }
  if(jlcxx::symbol_name(jl_current_module->name) == "Kokkos")
  {
    trilinoswrap::register_kokkos(registry.create_module("Kokkos"));
  }
  if(jlcxx::symbol_name(jl_current_module->name) == "Teuchos")
  {
    trilinoswrap::register_teuchos(registry.create_module("Teuchos"));
  }
  if(jlcxx::symbol_name(jl_current_module->name) == "Tpetra")
  {
    trilinoswrap::register_tpetra(registry.create_module("Tpetra"));
  }
  if(jlcxx::symbol_name(jl_current_module->name) == "Thyra")
  {
    trilinoswrap::register_thyra(registry.create_module("Thyra"));
  }

  if(jlcxx::symbol_name(jl_current_module->name) == "Benchmark")
  {
    trilinoswrap::register_benchmark(registry.create_module("Benchmark"));
  }
  if(jlcxx::symbol_name(jl_current_module->name) == "Testing")
  {
    trilinoswrap::register_testing(registry.create_module("Testing"));
  }
JULIA_CPP_MODULE_END
