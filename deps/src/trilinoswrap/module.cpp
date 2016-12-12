#include <cxx_wrap.hpp>

namespace trilinoswrap
{

extern void register_teuchos(cxx_wrap::Module mod);

} // namespace trilinoswrap


JULIA_CPP_MODULE_BEGIN(registry)
  using namespace cxx_wrap;

  trilinoswrap::register_teuchos(registry.create_module("Teuchos"));
JULIA_CPP_MODULE_END
