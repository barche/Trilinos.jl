#include <mpi.h>

#include "muelu.hpp"
#include "kokkos.hpp"
#include "teuchos.hpp"

#include <MueLu_CreateTpetraPreconditioner.hpp>

#include "jlcxx/jlcxx.hpp"

namespace trilinoswrap
{

struct WrapHierarchy
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.method("GetNumLevels", &WrappedT::GetNumLevels);
  }
};

struct WrapMueLuTpetraOperator
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.method("GetHierarchy", &WrappedT::GetHierarchy);
  }
};

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
struct MueLuParameters
{
  using ST = Scalar;
  using LT = LocalOrdinal;
  using GT = GlobalOrdinal;
  using NT = Node;
};

struct WrapMueLuFunctions
{
  WrapMueLuFunctions(jlcxx::Module& mod) : m_module(mod)
  {
  }

  template<typename ParametersT>
  void operator()()
  {
    typedef typename ParametersT::ST Scalar;
    typedef typename ParametersT::LT LocalOrdinal;
    typedef typename ParametersT::GT GlobalOrdinal;
    typedef typename ParametersT::NT Node;

    m_module.method("CreateTpetraPreconditioner", [] (const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &inA,
                         Teuchos::ParameterList& inParamList,
                         const Teuchos::RCP<Tpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node>>& inCoords,
                         const Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& inNullspace)
    {
      return MueLu::CreateTpetraPreconditioner(inA, inParamList, inCoords, inNullspace);
    });

    m_module.method("CreateTpetraPreconditioner", [] (const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &inA,
                         Teuchos::ParameterList& inParamList,
                         const Teuchos::RCP<Tpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node>>& inCoords)
    {
      return MueLu::CreateTpetraPreconditioner(inA, inParamList, inCoords);
    });

    m_module.method("CreateTpetraPreconditioner", [] (const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &inA,
                         Teuchos::ParameterList& inParamList)
    {
      return MueLu::CreateTpetraPreconditioner(inA, inParamList);
    });
  }

  jlcxx::Module& m_module;
};

void register_muelu(jlcxx::Module& mod)
{
  using namespace jlcxx;

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("Hierarchy")
    .apply_combination<MueLu::Hierarchy, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapHierarchy());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("TpetraOperator")
    .apply_combination<MueLu::TpetraOperator, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapMueLuTpetraOperator());

  typedef jlcxx::combine_types<jlcxx::ApplyType<MueLuParameters>, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t> params_t;
  jlcxx::for_each_type<params_t>(WrapMueLuFunctions(mod));
}

} // namespace trilinoswrap
