#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Teuchos_DefaultMpiComm.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

#include "teuchos.hpp"

namespace cxx_wrap
{

template<template<typename, typename, typename, bool> class T, typename P1, typename P2, typename P3, bool B>
struct BuildParameterList<T<P1,P2,P3,B>>
{
  typedef ParameterList<P1,P2,P3,bool> type;
};

template<template<typename, typename, typename, typename, bool> class T, typename P1, typename P2, typename P3, typename P4, bool B>
struct BuildParameterList<T<P1,P2,P3,P4,B>>
{
  typedef ParameterList<P1,P2,P3,P4,bool> type;
};

template<> struct IsBits<Tpetra::ProfileType> : std::true_type {};

template<typename T1, typename T2, typename T3, typename T4> struct CopyConstructible<Tpetra::Vector<T1,T2,T3,T4>> : std::false_type {};

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
    wrapped.method("getNodeNumElements", &WrappedT::getNodeNumElements);
    wrapped.method("getGlobalElement", &WrappedT::getGlobalElement);
    wrapped.method("isNodeGlobalElement", &WrappedT::isNodeGlobalElement);
  }
};

struct WrapCrsGraph
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::local_ordinal_type local_ordinal_type;
    typedef typename WrappedT::global_ordinal_type global_ordinal_type;

    wrapped.method("insertGlobalIndices",
      static_cast<void (WrappedT::*)(const global_ordinal_type, const Teuchos::ArrayView<const global_ordinal_type>&)>(&WrappedT::insertGlobalIndices));
    wrapped.module().method("fillComplete", [](WrappedT& w) { w.fillComplete(); });
    wrapped.method("getDomainMap", &WrappedT::getDomainMap);
    wrapped.method("getRangeMap", &WrappedT::getRangeMap);
    wrapped.method("getRowMap", &WrappedT::getRowMap);
    wrapped.module().method("resumeFill", [](WrappedT& w) { w.resumeFill(); });
    wrapped.method("getNumEntriesInGlobalRow", &WrappedT::getNumEntriesInGlobalRow);
    wrapped.method("getGlobalRowCopy", &WrappedT::getGlobalRowCopy);

    wrapped.module().method("CrsGraph", [](const Teuchos::RCP<const typename WrappedT::map_type>& rowmap, const std::size_t max_num_entries_per_row)
    {
      return Teuchos::rcp(new WrappedT(rowmap, max_num_entries_per_row, Tpetra::DynamicProfile));
    });
    wrapped.module().method("CrsGraph", [](const Teuchos::RCP<const typename WrappedT::map_type>& rowmap, const std::size_t max_num_entries_per_row, const Tpetra::ProfileType pftype)
    {
      return Teuchos::rcp(new WrappedT(rowmap, max_num_entries_per_row, pftype));
    });
  }
};

// Wrap the template type CrsMatrix<>
struct WrapCrsMatrix
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::scalar_type scalar_type;
    typedef typename WrappedT::local_ordinal_type local_ordinal_type;
    typedef typename WrappedT::global_ordinal_type global_ordinal_type;
    typedef typename WrappedT::node_type node_type;
    typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type> vector_type;

    wrapped.method("insertGlobalValues",
      static_cast<void (WrappedT::*)(const global_ordinal_type, const Teuchos::ArrayView<const global_ordinal_type>&, const Teuchos::ArrayView<const scalar_type>&)>(&WrappedT::insertGlobalValues));
    wrapped.module().method("fillComplete", [](WrappedT& w) { w.fillComplete(); });
    wrapped.method("getDomainMap", &WrappedT::getDomainMap);
    wrapped.method("getRangeMap", &WrappedT::getRangeMap);
    wrapped.method("getRowMap", &WrappedT::getRowMap);
    wrapped.method("apply", &WrappedT::apply);
    wrapped.module().method("apply", [] (const WrappedT& w, const vector_type& a, vector_type& b) { w.apply(a,b); });
    wrapped.module().method("resumeFill", [](WrappedT& w) { w.resumeFill(); });
    wrapped.method("getNumEntriesInGlobalRow", &WrappedT::getNumEntriesInGlobalRow);
    wrapped.method("getGlobalRowCopy", &WrappedT::getGlobalRowCopy);
    wrapped.method("replaceGlobalValues", static_cast<local_ordinal_type (WrappedT::*)(const global_ordinal_type, const Teuchos::ArrayView<const global_ordinal_type>&, const Teuchos::ArrayView<const scalar_type>&) const>(&WrappedT::replaceGlobalValues));
    wrapped.method("getFrobeniusNorm", &WrappedT::getFrobeniusNorm);

    wrapped.module().method("CrsMatrix", [](const Teuchos::RCP<const typename WrappedT::map_type>& rowmap, const std::size_t max_num_entries_per_row)
    {
      return Teuchos::rcp(new WrappedT(rowmap, max_num_entries_per_row, Tpetra::DynamicProfile));
    });
    wrapped.module().method("CrsMatrix", [](const Teuchos::RCP<const typename WrappedT::map_type>& rowmap, const std::size_t max_num_entries_per_row, const Tpetra::ProfileType pftype)
    {
      return Teuchos::rcp(new WrappedT(rowmap, max_num_entries_per_row, pftype));
    });
    wrapped.module().method("CrsMatrix", [](const Teuchos::RCP<const Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type>>& graph)
    {
      return Teuchos::rcp(new WrappedT(graph));
    });
    wrapped.module().method("convert", convert<WrappedT, Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type>>);
  }
};

struct WrapMultiVector
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::scalar_type scalar_type;
    typedef typename WrappedT::global_ordinal_type global_ordinal_type;

    wrapped.method("randomize", static_cast<void (WrappedT::*)()>(&WrappedT::randomize));
    wrapped.method("scale", static_cast<void (WrappedT::*)(const scalar_type&)>(&WrappedT::scale));
    wrapped.method("scale", static_cast<void (WrappedT::*)(const scalar_type&, const WrappedT&)>(&WrappedT::scale));
    wrapped.method("update", static_cast<void (WrappedT::*)(const scalar_type&, const WrappedT&, const scalar_type&)>(&WrappedT::update));
    wrapped.method("update", static_cast<void (WrappedT::*)(const scalar_type&, const WrappedT&, const scalar_type&, const WrappedT&, const scalar_type&)>(&WrappedT::update));
  }
};

struct WrapVector
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::scalar_type scalar_type;
    typedef typename WrappedT::global_ordinal_type global_ordinal_type;
    typedef typename WrappedT::mag_type mag_type;
    typedef typename WrappedT::map_type map_type;
    typedef typename WrappedT::dot_type dot_type;

    wrapped.method("norm2", static_cast<mag_type (WrappedT::*)() const>(&WrappedT::norm2));
    wrapped.method("dot", static_cast<dot_type (WrappedT::*)(const WrappedT&) const>(&WrappedT::dot));

    wrapped.module().method("Vector", [] (const Teuchos::RCP<const map_type>& map) { return Teuchos::rcp(new WrappedT(map)); });
    wrapped.module().method("convert", convert_unwrap<WrappedT, Tpetra::MultiVector<scalar_type, typename WrappedT::local_ordinal_type, global_ordinal_type>>);
  }
};

struct WrapTpetraNoOp
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&&)
  {
  }
};

void register_tpetra(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  mod.method("version", Tpetra::version);

  mod.add_bits<Tpetra::ProfileType>("ProfileType");
  mod.set_const("StaticProfile", Tpetra::StaticProfile);
  mod.set_const("DynamicProfile", Tpetra::DynamicProfile);

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("Map", rcp_wrappable())
    .apply<Tpetra::Map<int,int>, Tpetra::Map<int,int64_t>>(WrapMap());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("CrsGraph", rcp_wrappable())
    .apply<Tpetra::CrsGraph<int,int>, Tpetra::CrsGraph<int, int64_t>>(WrapCrsGraph());

  auto operator_type = mod.add_abstract<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("Operator", rcp_wrappable());
  operator_type.apply<Tpetra::Operator<double,int,int>, Tpetra::Operator<double,int,int64_t>>(WrapTpetraNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("CrsMatrix", operator_type.dt())
    .apply<Tpetra::CrsMatrix<double,int,int>, Tpetra::CrsMatrix<double,int,int64_t>>(WrapCrsMatrix());

  auto multivector = mod.add_abstract<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("MultiVector", rcp_wrappable());
  multivector.apply<Tpetra::MultiVector<double,int,int>, Tpetra::MultiVector<double,int,int64_t>>(WrapMultiVector());
  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("Vector", multivector.dt())
    .apply<Tpetra::Vector<double,int,int>, Tpetra::Vector<double,int,int64_t>>(WrapVector());

}

} // namespace trilinoswrap
