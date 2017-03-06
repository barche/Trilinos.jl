#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Thyra_BelosLinearOpWithSolveFactory.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>

#include "kokkos.hpp"
#include "teuchos.hpp"

namespace cxx_wrap
{

// heap-allocate solver result
template<typename T>
struct ConvertToJulia<Thyra::SolveStatus<T>, false, false, false>
{
  inline jl_value_t* operator()(const Thyra::SolveStatus<T>& cpp_obj) const
  {
    return create<Thyra::SolveStatus<T>>(cpp_obj);
  }
};

template<> struct IsBits<Thyra::EOpTransp> : std::true_type {};

}

namespace trilinoswrap
{

template<typename T>
struct WrapTpetraLinOpInternal;

template<typename ScalarT, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
struct WrapTpetraLinOpInternal<Thyra::TpetraLinearOp<ScalarT,LocalOrdinal,GlobalOrdinal,Node>>
{
  void operator()(cxx_wrap::Module& mod)
  {
    mod.method("tpetraVectorSpace", Thyra::tpetraVectorSpace<ScalarT,LocalOrdinal,GlobalOrdinal,Node>);
    mod.method("tpetraLinearOp", Thyra::tpetraLinearOp<ScalarT,LocalOrdinal,GlobalOrdinal,Node>);
    mod.method("tpetraVector", Thyra::tpetraVector<ScalarT,LocalOrdinal,GlobalOrdinal,Node>);
    mod.method("convert", convert<Thyra::TpetraVectorSpace<ScalarT,LocalOrdinal,GlobalOrdinal,Node>, Thyra::VectorSpaceBase<ScalarT>>);
    mod.method("convert", convert<Thyra::TpetraLinearOp<ScalarT,LocalOrdinal,GlobalOrdinal,Node>, Thyra::LinearOpBase<ScalarT>>);
  }
};

struct WrapTpetraLinOp
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    WrapTpetraLinOpInternal<WrappedT>()(wrapped.module());
  }
};

template<typename T>
struct extract_scalar_type;

template<template<typename, typename...> class T, typename Scalar, typename... OtherTs>
struct extract_scalar_type<T<Scalar, OtherTs...>>
{
  typedef Scalar type;
};

struct WrapSolveStatus
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename extract_scalar_type<WrappedT>::type Scalar;
  }
};

struct WrapLOWSFactory
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename extract_scalar_type<WrappedT>::type Scalar;
    wrapped.module().method("BelosLinearOpWithSolveFactory", [] (cxx_wrap::SingletonType<Scalar>) { return Teuchos::RCP<WrappedT>(new Thyra::BelosLinearOpWithSolveFactory<Scalar>()); });
    wrapped.module().method("setVerbLevel", [] (const WrappedT& w, const Teuchos::EVerbosityLevel level) { w.setVerbLevel(level); });
    wrapped.method("createOp", &WrappedT::createOp);
    wrapped.method("setParameterList", &WrappedT::setParameterList);
    wrapped.module().method("initializeOp", [](const Thyra::LinearOpWithSolveFactoryBase<Scalar>& lowsFactory, const Teuchos::RCP<const Thyra::LinearOpBase<Scalar> >& fwdOp, const Teuchos::RCP<Thyra::LinearOpWithSolveBase<Scalar> >& Op)
    {
      Thyra::initializeOp(lowsFactory, fwdOp, Op.ptr());
    });
    wrapped.module().method("solve", Thyra::solve<Scalar>);
    wrapped.module().method("solve", [] (const Thyra::LinearOpWithSolveBase<Scalar>& lows, const Thyra::EOpTransp trans, const Thyra::MultiVectorBase<Scalar>& rhs, Teuchos::Ptr<Thyra::MultiVectorBase<Scalar>> x)
    {
      return Thyra::solve(lows, trans, rhs, x);
    });
  }
};

struct WrapTpetraVector
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename extract_scalar_type<WrappedT>::type Scalar;
    wrapped.module().method("convert", convert<WrappedT, Thyra::MultiVectorBase<Scalar>>);
  }
};

struct WrapNoOp
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
  }
};

void register_thyra(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  auto vecspace_base = mod.add_abstract<Parametric<TypeVar<1>>>("VectorSpaceBase", rcp_wrappable());
  vecspace_base.apply_combination<Thyra::VectorSpaceBase, scalars_t>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>, ParameterList<TypeVar<1>>>("TpetraVectorSpace", vecspace_base.dt())
    .apply_combination<Thyra::TpetraVectorSpace, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapNoOp());

  auto multi_vector_base = mod.add_abstract<Parametric<TypeVar<1>>>("MultiVectorBase", rcp_wrappable());
  multi_vector_base.apply_combination<Thyra::MultiVectorBase, scalars_t>(WrapNoOp());

  auto vector_base = mod.add_abstract<Parametric<TypeVar<1>>>("VectorBase", multi_vector_base.dt());
  vector_base.apply_combination<Thyra::VectorBase, scalars_t>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>, ParameterList<TypeVar<1>>>("TpetraVector", vector_base.dt())
    .apply_combination<Thyra::TpetraVector, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapTpetraVector());

  auto linop_base = mod.add_abstract<Parametric<TypeVar<1>>>("LinearOpBase", rcp_wrappable());
  linop_base.apply_combination<Thyra::LinearOpBase, scalars_t>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>>>("LinearOpWithSolveBase", rcp_wrappable())
    .apply_combination<Thyra::LinearOpWithSolveBase, scalars_t>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>, ParameterList<TypeVar<1>>>("TpetraLinearOp", linop_base.dt())
    .apply_combination<Thyra::TpetraLinearOp, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapTpetraLinOp());

  mod.add_bits<Thyra::EOpTransp>("EOpTransp");
  mod.set_const("NOTRANS", Thyra::NOTRANS);
  mod.set_const("CONJ", Thyra::CONJ);
  mod.set_const("TRANS", Thyra::TRANS);
  mod.set_const("CONJTRANS", Thyra::CONJTRANS);

  mod.add_type<Parametric<TypeVar<1>>>("SolveStatus")
    .apply_combination<Thyra::SolveStatus, scalars_t>(WrapSolveStatus());

  mod.add_type<Parametric<TypeVar<1>>>("SolveCriteria", rcp_wrappable())
    .apply_combination<Thyra::SolveCriteria, scalars_t>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>>>("LinearOpWithSolveFactoryBase", rcp_wrappable())
    .apply_combination<Thyra::LinearOpWithSolveFactoryBase, scalars_t>(WrapLOWSFactory());
}

} // namespace trilinoswrap
