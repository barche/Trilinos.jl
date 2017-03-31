#include <cxx_wrap.hpp>
#include <mpi.h>

#include "teuchos.hpp"
#include "tpetra.hpp"

#include <Teuchos_ArrayView.hpp>

namespace trilinoswrap
{

template<typename T>
int arrayview_sum(const Teuchos::ArrayView<T>& av)
{
  T result = 0;
  for(T x : av)
  {
    result += x;
  }
  return result;
}

// Some methods to test types that would otherwise be difficult to test directly
void register_testing(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  mod.method("arrayview_sum", arrayview_sum<double>);
}

} // namespace trilinoswrap
