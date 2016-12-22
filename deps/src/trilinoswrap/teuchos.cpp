#include <cxx_wrap.hpp>
#include <mpi.h>

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

  template<>
  struct ConvertToCpp<MPI_Comm, false, false, true>
  {
    inline MPI_Comm operator()(MPI_Comm julia_value) const
    {
      return julia_value;
    }
  };
}

namespace trilinoswrap
{

jl_datatype_t* g_rcp_type;

void register_teuchos(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  mod.add_type<Parametric<TypeVar<1>>>("RCP");
  g_rcp_type = mod.get_julia_type("RCP");

  mod.add_abstract<Teuchos::Comm<int>>("Comm");
  mod.add_type<Teuchos::MpiComm<int>>("MpiComm", julia_type<Teuchos::Comm<int>>());
  mod.method("MpiComm", [](MPI_Comm comm)
  {
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm(new Teuchos::MpiComm<int>(comm));
    return teuchos_comm;
  });

  mod.method("getRank", [](const Teuchos::RCP<const Teuchos::Comm<int>>& c) { return c->getRank(); });
  mod.method("getSize", [](const Teuchos::RCP<const Teuchos::Comm<int>>& c) { return c->getSize(); });
}

} // namespace trilinoswrap
