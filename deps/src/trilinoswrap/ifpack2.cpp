#include "jlcxx/jlcxx.hpp"
#include <mpi.h>

#include "ifpack2.hpp"
#include "kokkos.hpp"
#include "teuchos.hpp"

namespace trilinoswrap
{

struct WrapPreconditioner
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::scalar_type ST;
    typedef typename WrappedT::local_ordinal_type LT;
    typedef typename WrappedT::global_ordinal_type GT;
    typedef typename WrappedT::node_type NT;

    wrapped.method("setParameters", &WrappedT::setParameters);
    wrapped.method("initialize", &WrappedT::initialize);
    wrapped.method("compute", &WrappedT::compute);

    wrapped.module().method("create", [](const Ifpack2::Factory& factory, const std::string& prec_type, const Teuchos::RCP<const Tpetra::CrsMatrix<ST,LT,GT,NT>>& mat)
    {
      return factory.create(prec_type, mat);
    });
  }
};

void register_ifpack2(jlcxx::Module& mod)
{
  using namespace jlcxx;

  mod.add_type<Ifpack2::Factory>("Factory");

  // mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("Preconditioner", jlcxx::julia_type("Operator", "Tpetra"))
  //   .apply_combination<Ifpack2::Preconditioner, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapPreconditioner());
}

} // namespace trilinoswrap
