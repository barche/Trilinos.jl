#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

#include "kokkos.hpp"
#include "teuchos.hpp"
#include "tpetra.hpp"

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
    typedef typename WrappedT::node_type NodeT;
    wrapped.module().method("Map", [](const Tpetra::global_size_t num_indices, const cxx_wrap::StrictlyTypedNumber<typename WrappedT::global_ordinal_type> index_base, const Teuchos::RCP<const Teuchos::Comm<int>>& comm, cxx_wrap::SingletonType<NodeT>)
    {
      return Teuchos::rcp(new WrappedT(num_indices, index_base.value, comm));
    });
    wrapped.method("getNodeNumElements", &WrappedT::getNodeNumElements);
    wrapped.method("getGlobalElement", &WrappedT::getGlobalElement);
    wrapped.method("getLocalElement", &WrappedT::getLocalElement);
    wrapped.method("isNodeGlobalElement", &WrappedT::isNodeGlobalElement);
    wrapped.method("getIndexBase", &WrappedT::getIndexBase);
    wrapped.method("getMinLocalIndex", &WrappedT::getMinLocalIndex);
    wrapped.method("getMaxLocalIndex", &WrappedT::getMaxLocalIndex);
    wrapped.method("getMinGlobalIndex", &WrappedT::getMinGlobalIndex);
    wrapped.method("getMaxGlobalIndex", &WrappedT::getMaxGlobalIndex);
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
    typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> vector_type;

    wrapped.method("insertGlobalValues",
      static_cast<void (WrappedT::*)(const global_ordinal_type, const Teuchos::ArrayView<const global_ordinal_type>&, const Teuchos::ArrayView<const scalar_type>&)>(&WrappedT::insertGlobalValues));
    wrapped.module().method("fillComplete", [](WrappedT& w) { w.fillComplete(); });
    wrapped.method("getDomainMap", &WrappedT::getDomainMap);
    wrapped.method("getRangeMap", &WrappedT::getRangeMap);
    wrapped.method("getRowMap", &WrappedT::getRowMap);
    wrapped.method("_apply", &WrappedT::apply); // Default arguments not set, hence the _
    wrapped.module().method("apply", [] (const WrappedT& w, const vector_type& a, vector_type& b) { w.apply(a,b); });
    wrapped.module().method("resumeFill", [](WrappedT& w) { w.resumeFill(); });
    wrapped.method("getNumEntriesInGlobalRow", &WrappedT::getNumEntriesInGlobalRow);
    wrapped.method("getGlobalRowCopy", &WrappedT::getGlobalRowCopy);
    wrapped.method("replaceGlobalValues", static_cast<local_ordinal_type (WrappedT::*)(const global_ordinal_type, const Teuchos::ArrayView<const global_ordinal_type>&, const Teuchos::ArrayView<const scalar_type>&) const>(&WrappedT::replaceGlobalValues));
    wrapped.method("replaceLocalValues", static_cast<local_ordinal_type (WrappedT::*)(const local_ordinal_type, const Teuchos::ArrayView<const local_ordinal_type>&, const Teuchos::ArrayView<const scalar_type>&) const>(&WrappedT::replaceLocalValues));
    wrapped.method("getFrobeniusNorm", &WrappedT::getFrobeniusNorm);
    wrapped.method("apply", &WrappedT::apply);
    wrapped.method("isLocallyIndexed", &WrappedT::isLocallyIndexed);
    wrapped.method("isGloballyIndexed", &WrappedT::isGloballyIndexed);

    wrapped.module().method("CrsMatrix", [](const Teuchos::RCP<const typename WrappedT::map_type>& rowmap, const std::size_t max_num_entries_per_row)
    {
      return Teuchos::rcp(new WrappedT(rowmap, max_num_entries_per_row, Tpetra::DynamicProfile));
    });
    wrapped.module().method("CrsMatrix", [](const Teuchos::RCP<const typename WrappedT::map_type>& rowmap, const std::size_t max_num_entries_per_row, const Tpetra::ProfileType pftype)
    {
      return Teuchos::rcp(new WrappedT(rowmap, max_num_entries_per_row, pftype));
    });
    wrapped.module().method("CrsMatrix", [](const Teuchos::RCP<const Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type>>& graph)
    {
      return Teuchos::rcp(new WrappedT(graph));
    });
    wrapped.module().method("describe", [](const WrappedT& mat, const Teuchos::EVerbosityLevel verbLevel) { mat.describe(*Teuchos::VerboseObjectBase::getDefaultOStream(), verbLevel); });
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
    typedef typename WrappedT::map_type map_type;
    typedef typename WrappedT::dot_type dot_type;

    wrapped.method("randomize", static_cast<void (WrappedT::*)()>(&WrappedT::randomize));
    wrapped.method("scale", static_cast<void (WrappedT::*)(const scalar_type&)>(&WrappedT::scale));
    wrapped.method("scale", static_cast<void (WrappedT::*)(const scalar_type&, const WrappedT&)>(&WrappedT::scale));
    wrapped.method("update", static_cast<void (WrappedT::*)(const scalar_type&, const WrappedT&, const scalar_type&)>(&WrappedT::update));
    wrapped.method("update", static_cast<void (WrappedT::*)(const scalar_type&, const WrappedT&, const scalar_type&, const WrappedT&, const scalar_type&)>(&WrappedT::update));
    wrapped.method("putScalar", static_cast<void (WrappedT::*)(const scalar_type&)>(&WrappedT::putScalar));
    wrapped.method("dot", static_cast<void (WrappedT::*)(const WrappedT&, const Teuchos::ArrayView<dot_type>&) const>(&WrappedT::dot));

    wrapped.method("getMap", &WrappedT::getMap);

    wrapped.module().method("MultiVector", [] (const Teuchos::RCP<const map_type>& map, const std::size_t num_vecs) { return Teuchos::rcp(new WrappedT(map, num_vecs)); });

    typedef typename WrappedT::dual_view_type dual_view_type;
    typedef typename dual_view_type::t_host host_view_type;
    typedef typename dual_view_type::t_dev device_view_type;
    wrapped.module().method("host_view_type", [] (WrappedT& vec) { return cxx_wrap::SingletonType<host_view_type>(); });
    wrapped.module().method("device_view_type", [] (WrappedT& vec) { return cxx_wrap::SingletonType<device_view_type>(); });
    wrapped.module().method("getLocalView", [] (cxx_wrap::SingletonType<host_view_type>, WrappedT& vec) {return cxx_wrap::create<host_view_type>(vec.template getLocalView<host_view_type>()); } ).set_return_type(cxx_wrap::julia_type<host_view_type>());
    wrapped.module().method("getLocalView", [] (cxx_wrap::SingletonType<device_view_type>, WrappedT& vec) {return cxx_wrap::create<device_view_type>(vec.template getLocalView<device_view_type>()); } ).set_return_type(cxx_wrap::julia_type<device_view_type>());
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
    typedef typename WrappedT::node_type node_type;
    typedef typename WrappedT::mag_type mag_type;
    typedef typename WrappedT::map_type map_type;
    typedef typename WrappedT::dot_type dot_type;

    wrapped.method("norm2", static_cast<mag_type (WrappedT::*)() const>(&WrappedT::norm2));
    wrapped.method("dot", static_cast<dot_type (WrappedT::*)(const WrappedT&) const>(&WrappedT::dot));

    wrapped.module().method("Vector", [] (const Teuchos::RCP<const map_type>& map) { return Teuchos::rcp(new WrappedT(map)); });
  }
};

struct WrapTpetraNoOp
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&&)
  {
  }
};

// Allows using apply_combination on a type that has a non-type parameter (bool in this case, with a default  value we never change)
template<template<typename, typename, typename, bool> class T>
struct ApplyTpetra3
{
  template<typename... Types> using apply = T<Types...>;
};

template<template<typename, typename, typename, typename, bool> class T>
struct ApplyTpetra4
{
  template<typename... Types> using apply = T<Types...>;
};

void register_tpetra(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  mod.method("version", Tpetra::version);

  mod.add_bits<Tpetra::ProfileType>("ProfileType");
  mod.set_const("StaticProfile", Tpetra::StaticProfile);
  mod.set_const("DynamicProfile", Tpetra::DynamicProfile);

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("Map")
    .apply_combination<Tpetra::Map, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapMap());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("CrsGraph")
    .apply_combination<ApplyTpetra3<Tpetra::CrsGraph>, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapCrsGraph());

  auto operator_type = mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("Operator");
  operator_type.apply_combination<Tpetra::Operator, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapTpetraNoOp());

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("CrsMatrix", operator_type.dt())
    .apply_combination<ApplyTpetra4<Tpetra::CrsMatrix>, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapCrsMatrix());

  auto multivector = mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("MultiVector");
  multivector.apply_combination<ApplyTpetra4<Tpetra::MultiVector>, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapMultiVector());
  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("Vector", multivector.dt())
    .apply_combination<ApplyTpetra4<Tpetra::Vector>, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t>(WrapVector());
}

} // namespace trilinoswrap
