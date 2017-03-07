#include <cxx_wrap.hpp>
#include <mpi.h>

#include "kokkos.hpp"

#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_View.hpp>
#include <TpetraCore_config.h>

namespace trilinoswrap
{

struct WrapView
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::value_type ScalarT;
    wrapped.method("dimension", &WrappedT::template dimension<int_t>);
    wrapped.method(&WrappedT::template operator()<int_t, int_t>);
    wrapped.method("ptr_on_device", &WrappedT::ptr_on_device);
  }
};

void register_kokkos(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;
#ifdef HAVE_TPETRA_INST_SERIAL
  mod.add_type<Kokkos::Compat::KokkosSerialWrapperNode>("KokkosSerialWrapperNode");
  mod.add_type<Kokkos::Serial>("Serial");
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
  mod.add_type<Kokkos::Compat::KokkosOpenMPWrapperNode>("KokkosOpenMPWrapperNode");
  mod.add_type<Kokkos::OpenMP>("OpenMP");
#endif

  mod.method("default_node_type", [] () { return cxx_wrap::julia_type<KokkosClassic::DefaultNode::DefaultNodeType>(); });

  mod.add_type<Kokkos::LayoutLeft>("LayoutLeft");
  mod.add_type<Kokkos::LayoutRight>("LayoutRight");

  typedef ParameterList<Kokkos::LayoutLeft, Kokkos::LayoutRight> layouts_t;

  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>>>("View3")
    .apply_combination<Kokkos::View, arrays_t, layouts_t, kokkos_devices_t>(WrapView());
  mod.add_type<Parametric<TypeVar<1>, TypeVar<2>, TypeVar<3>, TypeVar<4>>>("View4")
    .apply_combination<Kokkos::View, arrays_t, layouts_t, kokkos_devices_t, ParameterList<void>>(WrapView());
}

} // namespace trilinoswrap
