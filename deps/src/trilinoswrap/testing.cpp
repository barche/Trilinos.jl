#include "jlcxx/jlcxx.hpp"
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
void register_testing(jlcxx::Module& mod)
{
  using namespace jlcxx;

  mod.method("arrayview_sum", arrayview_sum<double>);
}

} // namespace trilinoswrap
