#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Teuchos_DefaultMpiComm.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Version.hpp>

#include "teuchos.hpp"

namespace cxx_wrap
{

// Match the Eigen Matrix type, skipping the default parameters
template<typename LocalOrdinalT, typename GlobalOrdinalT>
struct BuildParameterList<Tpetra::Map<LocalOrdinalT, GlobalOrdinalT>>
{
  typedef ParameterList<LocalOrdinalT, GlobalOrdinalT> type;
};

}

namespace trilinoswrap
{

// Wrap the template type Map<>
struct WrapMap
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.module().method("Map", [](const Tpetra::global_size_t num_indices, const cxx_wrap::StrictlyTypedNumber<typename WrappedT::global_ordinal_type> index_base, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
    {
      return Teuchos::rcp(new WrappedT(num_indices, index_base.value, comm));
    });
    wrapped.module().method("getNodeNumElements", [] (const Teuchos::RCP<WrappedT const>& map) { return map->getNodeNumElements(); });
  }
};

void register_tpetra(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  mod.method("version", Tpetra::version);

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("Map")
    .apply<Tpetra::Map<int,int>, Tpetra::Map<int,int64_t>>(WrapMap());


}

} // namespace trilinoswrap
