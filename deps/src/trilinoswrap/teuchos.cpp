#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Teuchos_DefaultMpiComm.hpp>

namespace cxx_wrap
{
  // Some special-casing for RCP
  template<typename T>
  struct InstantiateParametricType<Teuchos::RCP<T>>
  {
    int operator()(Module& m) const
    {
      // Register the Julia type if not already instantiated
      if(!static_type_mapping<Teuchos::RCP<T>>::has_julia_type())
      {
        jl_datatype_t* dt = (jl_datatype_t*)jl_apply_type((jl_value_t*)m.get_julia_type("RCP"), jl_svec1(static_type_mapping<typename std::remove_const<T>::type>::julia_type()));
        set_julia_type<Teuchos::RCP<T>>(dt);
        protect_from_gc(dt);
      }
      return 0;
    }
  };

  template<typename T>
  struct ConvertToJulia<Teuchos::RCP<T>, false, false, false>
  {
    jl_value_t* operator()(const Teuchos::RCP<T>& cpp_obj) const
    {
      return create<Teuchos::RCP<T>>(cpp_obj);
    }
  };

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

void register_teuchos(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  mod.add_type<Parametric<TypeVar<1>>>("RCP");

  mod.add_abstract<Teuchos::Comm<int>>("Comm");
  mod.add_type<Teuchos::MpiComm<int>>("MpiComm", julia_type<Teuchos::Comm<int>>());
  mod.method("MpiComm", [](MPI_Comm comm)
  {
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm(new Teuchos::MpiComm<int>(comm));
    return teuchos_comm;
  });

  mod.method("getrank", [](const Teuchos::RCP<const Teuchos::Comm<int>>& c) { return c->getRank(); });
  mod.method("getsize", [](const Teuchos::RCP<const Teuchos::Comm<int>>& c) { return c->getSize(); });
}

} // namespace trilinoswrap
