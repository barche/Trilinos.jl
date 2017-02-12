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

#include "teuchos.hpp"

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

template<template<typename> class T, typename Scalar>
struct extract_scalar_type<T<Scalar>>
{
  typedef Scalar type;
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

  auto vecspace_base = mod.add_type<Parametric<TypeVar<1>>>("VectorSpaceBase", rcp_wrappable())
    .apply<Thyra::VectorSpaceBase<double>, Thyra::VectorSpaceBase<float>>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("TpetraVectorSpace", vecspace_base.dt())
    .apply<Thyra::TpetraVectorSpace<double,int,int,KokkosClassic::DefaultNode::DefaultNodeType>, Thyra::TpetraVectorSpace<double,int,int64_t,KokkosClassic::DefaultNode::DefaultNodeType>>(WrapNoOp());

  auto vector_base = mod.add_type<Parametric<TypeVar<1>>>("VectorBase", rcp_wrappable());
  vector_base.apply<Thyra::VectorBase<double>, Thyra::VectorBase<float>>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("TpetraVector", vector_base.dt())
    .apply<Thyra::TpetraVector<double,int,int,KokkosClassic::DefaultNode::DefaultNodeType>, Thyra::TpetraVector<double,int,int64_t,KokkosClassic::DefaultNode::DefaultNodeType>>(WrapNoOp());

  auto linop_base = mod.add_type<Parametric<TypeVar<1>>>("LinearOpBase", rcp_wrappable())
    .apply<Thyra::LinearOpBase<double>, Thyra::LinearOpBase<float>>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>>>("LinearOpWithSolveBase", rcp_wrappable())
    .apply<Thyra::LinearOpWithSolveBase<double>, Thyra::LinearOpWithSolveBase<float>>(WrapNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("TpetraLinearOp", linop_base.dt())
    .apply<Thyra::TpetraLinearOp<double,int,int>, Thyra::TpetraLinearOp<double,int,int64_t>>(WrapTpetraLinOp());

  mod.add_type<Parametric<TypeVar<1>>>("LinearOpWithSolveFactoryBase", rcp_wrappable())
    .apply<Thyra::LinearOpWithSolveFactoryBase<double>, Thyra::LinearOpWithSolveFactoryBase<float>>(WrapLOWSFactory());
}

} // namespace trilinoswrap
