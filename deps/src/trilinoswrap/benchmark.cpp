#include "jlcxx/jlcxx.hpp"
#include <mpi.h>

#include <Teuchos_DefaultMpiComm.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include "teuchos.hpp"
#include "tpetra.hpp"

namespace trilinoswrap
{

void register_benchmark(jlcxx::Module& mod)
{
  using namespace jlcxx;

  mod.method("vector_fill", [] (const Teuchos::RCP<Tpetra::Map<int, int64_t>>& map, const Teuchos::RCP<Tpetra::Vector<double,int,int64_t>>& v)
  {
    const size_t num_my_elements = map->getNodeNumElements();
    auto dev_view = v->getLocalView<Tpetra::Vector<double,int,int64_t>::dual_view_type::t_dev>();
    for(size_t i = 0; i != num_my_elements; ++i)
    {
      dev_view(i,0) = i;
    }
  });

  mod.method("multivector_fill", [] (const Teuchos::RCP<Tpetra::Map<int, int64_t>>& map, const Teuchos::RCP<Tpetra::MultiVector<double,int,int64_t>>& v)
  {
    const size_t num_my_elements = map->getNodeNumElements();
    auto dev_view = v->getLocalView<Tpetra::Vector<double,int,int64_t>::dual_view_type::t_dev>();
    const size_t num_vectors = v->getNumVectors();
    for(size_t j = 0; j != num_vectors; ++j)
    {
      for(size_t i = 0; i != num_my_elements; ++i)
      {
        dev_view(i,j) = i*(j+1);
      }
    }
  });
}

} // namespace trilinoswrap
