#include <mpi.h>

#include "kokkos.hpp"

#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_StaticCrsGraph.hpp>
#include <Kokkos_View.hpp>
#include <TpetraCore_config.h>

#include "jlcxx/jlcxx.hpp"

namespace trilinoswrap
{

struct WrapDevice
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&&)
  {
  }
};

namespace detail
{

  template<typename OutViewT, typename ViewT, typename DT>
  OutViewT make_view(jlcxx::SingletonType<OutViewT>, const std::string& name, jlcxx::ArrayRef<DT> arr)
  {
    const std::size_t length = arr.size();
    ViewT newview(name, length);
    auto hostview = Kokkos::create_mirror_view(newview);
    for(std::size_t i = 0; i != length; ++i)
    {
      hostview(i) = arr[i];
    }
    Kokkos::deep_copy(newview, hostview);
    return newview;
  }

  template<int Rank, typename ViewT>
  struct ViewConstructor
  {
    void operator()(jlcxx::Module&)
    {
    }
  };

  template<int Rank, typename DT, typename AT1, typename AT2>
  struct ViewConstructor<Rank, Kokkos::View<const DT*, AT1, AT2>>
  {
    void operator()(jlcxx::Module&)
    {
    }
  };

  template<typename DT, typename AT1, typename AT2>
  struct ViewConstructor<1, Kokkos::View<const DT*, AT1, AT2>>
  {
    typedef Kokkos::View<DT*, AT1, AT2> ViewT;
    typedef Kokkos::View<const DT*, AT1, AT2> ConstViewT;

    void operator()(jlcxx::Module& mod)
    {
      mod.method("makeview", make_view<ConstViewT,ViewT,DT>);
    }
  };

  template<typename DT, typename AT1, typename AT2>
  struct ViewConstructor<1, Kokkos::View<DT*, AT1, AT2>>
  {
    typedef Kokkos::View<DT*, AT1, AT2> ViewT;

    void operator()(jlcxx::Module& mod)
    {
      mod.method("makeview", make_view<ViewT,ViewT,DT>);
    }
  };
}

struct WrapView
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename std::remove_const<typename WrappedT::value_type>::type ScalarT;
    wrapped.method("dimension", &WrappedT::template dimension<int_t>);
    wrapped.method(&WrappedT::template operator()<int_t, int_t>);
    wrapped.method("ptr_on_device", &WrappedT::ptr_on_device);
    wrapped.module().method("value_type", [] (jlcxx::SingletonType<WrappedT>) { return jlcxx::SingletonType<ScalarT>(); });
    wrapped.module().method("array_layout", [] (jlcxx::SingletonType<WrappedT>) { return jlcxx::SingletonType<typename WrappedT::array_layout>(); });
    wrapped.module().method("rank", [] (jlcxx::SingletonType<WrappedT>) { return int_t(WrappedT::rank); });
    detail::ViewConstructor<WrappedT::rank, WrappedT>()(wrapped.module());
  }
};



struct WrapStaticCrsGraph
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::entries_type entries_type;
    typedef typename WrappedT::row_map_type row_map_type;
    wrapped.template constructor<const entries_type&, const row_map_type>();
    wrapped.module().method("entries_type", [] (jlcxx::SingletonType<WrappedT>) { return jlcxx::SingletonType<entries_type>(); });
    wrapped.module().method("row_map_type", [] (jlcxx::SingletonType<WrappedT>) { return jlcxx::SingletonType<row_map_type>(); });
  }
};

void register_kokkos(jlcxx::Module& mod)
{
  using namespace jlcxx;
  auto device_type = mod.add_type<Parametric<TypeVar<1>,TypeVar<2>>>("Device");
  mod.add_type<Kokkos::HostSpace>("HostSpace");
#ifdef HAVE_TPETRA_INST_SERIAL
  mod.add_type<Kokkos::Compat::KokkosSerialWrapperNode>("KokkosSerialWrapperNode");
  mod.add_type<Kokkos::Serial>("Serial");
  device_type.apply<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(WrapDevice());
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
  mod.add_type<Kokkos::Compat::KokkosOpenMPWrapperNode>("KokkosOpenMPWrapperNode");
  mod.add_type<Kokkos::OpenMP>("OpenMP");
  mod.method("initialize", [](Kokkos::OpenMP, int num_threads) { Kokkos::OpenMP::initialize(num_threads); });
  device_type.apply<Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>(WrapDevice());
#endif

  mod.method("default_node_type", [] () { return (jl_datatype_t*)jlcxx::julia_type<KokkosClassic::DefaultNode::DefaultNodeType>(); });

  mod.add_type<Kokkos::LayoutLeft>("LayoutLeft");
  mod.add_type<Kokkos::LayoutRight>("LayoutRight");

  typedef ParameterList<Kokkos::LayoutLeft, Kokkos::LayoutRight> layouts_t;

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("View3")
    .apply_combination<Kokkos::View, arrays_t, layouts_t, kokkos_devices_t>(WrapView());
  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("View4")
    .apply_combination<Kokkos::View, arrays_t, layouts_t, kokkos_devices_t, ParameterList<void>>(WrapView());
  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("StaticCrsGraph_cpp")
    .apply_combination<Kokkos::StaticCrsGraph, local_ordinals_t, layouts_t, kokkos_devices_t>(WrapStaticCrsGraph());
}

} // namespace trilinoswrap
