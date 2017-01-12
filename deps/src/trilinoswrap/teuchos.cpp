#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include "teuchos.hpp"

namespace cxx_wrap
{
  // Support MPI.CComm directly
  template<> struct IsBits<MPI_Comm> : std::true_type {};

  template<> struct static_type_mapping<MPI_Comm>
  {
    typedef MPI_Comm type;
    static jl_datatype_t* julia_type() { return ::cxx_wrap::julia_type("CComm", "MPI"); }
    template<typename T> using remove_const_ref = cxx_wrap::remove_const_ref<T>;
  };

  template<> struct IsBits<Teuchos::ETransp> : std::true_type {};
}

namespace trilinoswrap
{

jl_datatype_t* g_rcp_type;

jl_datatype_t* julia_rcp_wrappable()
{
  return cxx_wrap::julia_type("JuliaRCPWrappable", "Trilinos");
}

struct WrapArrayView
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    typedef typename WrappedT::value_type value_type;
    wrapped.method("size", &WrappedT::size);
    wrapped.module().method("ArrayView", [](cxx_wrap::ArrayRef<value_type, 1> arr)
    {
      return cxx_wrap::create<WrappedT>(arr.data(), arr.size());
    });
  }
};

void register_teuchos(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  // RCP
  mod.add_type<Parametric<TypeVar<1>>>("RCP");
  g_rcp_type = mod.get_julia_type("RCP");

  // Comm
  mod.add_abstract<Teuchos::Comm<int>>("Comm", julia_rcp_wrappable())
    .method("getRank", &Teuchos::Comm<int>::getRank)
    .method("getSize", &Teuchos::Comm<int>::getSize);
  mod.add_type<Teuchos::MpiComm<int>>("MpiComm", julia_type<Teuchos::Comm<int>>());
  mod.method("MpiComm", [](MPI_Comm comm)
  {
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm(new Teuchos::MpiComm<int>(comm));
    return teuchos_comm;
  });

  mod.add_type<Parametric<TypeVar<1>>>("ArrayView")
    .apply<Teuchos::ArrayView<double>, Teuchos::ArrayView<int>, Teuchos::ArrayView<int64_t>>(WrapArrayView());

  mod.add_bits<Teuchos::ETransp>("ETransp");
  mod.set_const("NO_TRANS", Teuchos::NO_TRANS);
  mod.set_const("TRANS", Teuchos::TRANS);
  mod.set_const("CONJ_TRANS", Teuchos::CONJ_TRANS);
}

} // namespace trilinoswrap
