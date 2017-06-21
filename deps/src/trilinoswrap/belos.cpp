#include "jlcxx/jlcxx.hpp"
#include <mpi.h>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

#include "ifpack2.hpp"
#include "kokkos.hpp"
#include "teuchos.hpp"

namespace jlcxx
{
  // Disable default construction to force creation in RCP
  template<typename ST, typename MV, typename OP> struct DefaultConstructible<Belos::LinearProblem<ST,MV,OP>> : std::false_type {};

  template<> struct IsBits<Belos::ReturnType> : std::true_type {};
}

namespace trilinoswrap
{

template<typename T>
struct BelosTraits;

template<typename ST, typename MV, typename OP>
struct BelosTraits<Belos::LinearProblem<ST,MV,OP>>
{
  typedef ST scalar_type;
  typedef MV multivector_type;
  typedef OP operator_type;
};

struct WrapLinearProblem
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename BelosTraits<WrappedT>::operator_type OP;
    wrapped.module().method("LinearProblem", [] (const Teuchos::RCP<const OP>& op) { return Teuchos::rcp(new WrappedT(op, Teuchos::null, Teuchos::null)); });
    wrapped.method("setOperator", &WrappedT::setOperator);
    wrapped.method("getOperator", &WrappedT::getOperator);
    wrapped.method("setLeftPrec", &WrappedT::setLeftPrec);
    wrapped.method("setRightPrec", &WrappedT::setRightPrec);
    wrapped.method("setProblem", &WrappedT::setProblem);
    wrapped.method("setLHS", &WrappedT::setLHS);
    wrapped.method("setRHS", &WrappedT::setRHS);
  }
};

struct WrapSolverManager
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.method("getProblem", &WrappedT::getProblem);
    wrapped.method("setProblem", &WrappedT::setProblem);
    wrapped.method("solve", &WrappedT::solve);
  }
};

struct WrapSolverFactory
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.method("create", &WrappedT::create);
  }
};

struct ApplyLinearProblem
{
  template<typename ST, typename LT, typename GT, typename NT>
  using apply = Belos::LinearProblem<ST, Tpetra::MultiVector<ST,LT,GT,NT>, Tpetra::Operator<ST,LT,GT,NT>>;
};

struct ApplySolverManager
{
  template<typename ST, typename LT, typename GT, typename NT>
  using apply = Belos::SolverManager<ST, Tpetra::MultiVector<ST,LT,GT,NT>, Tpetra::Operator<ST,LT,GT,NT>>;
};

struct ApplySolverFactory
{
  template<typename ST, typename LT, typename GT, typename NT>
  using apply = Belos::SolverFactory<ST, Tpetra::MultiVector<ST,LT,GT,NT>, Tpetra::Operator<ST,LT,GT,NT>>;
};

void register_belos(jlcxx::Module& mod)
{
  using namespace jlcxx;

  mod.add_bits<Belos::ReturnType>("ReturnType");
  mod.set_const("Converged", Belos::Converged);
  mod.set_const("Unconverged", Belos::Unconverged);

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("LinearProblem")
    .apply_combination<ApplyLinearProblem, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapLinearProblem());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("SolverManager")
    .apply_combination<ApplySolverManager, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapSolverManager());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("SolverFactory")
    .apply_combination<ApplySolverFactory, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapSolverFactory());
}

} // namespace trilinoswrap
