#ifndef TRILINOS_JL_RCP_HPP
#define TRILINOS_JL_RCP_HPP

#include <cxx_wrap.hpp>
#include <Teuchos_RCP.hpp>

namespace trilinoswrap
{
  extern jl_datatype_t* g_rcp_type;

  jl_datatype_t* julia_rcp_type();
}

namespace cxx_wrap
{

// Some special-casing for RCP
template<typename T>
struct InstantiateParametricType<Teuchos::RCP<T>>
{
  inline int operator()(Module& m) const
  {
    // Register the Julia type if not already instantiated
    if(!static_type_mapping<Teuchos::RCP<T>>::has_julia_type())
    {
      typedef typename std::remove_const<T>::type nonconst_t;
      jl_datatype_t* wrapped_t = static_type_mapping<nonconst_t>::julia_type();

      jl_datatype_t* dt = (jl_datatype_t*)jl_apply_type((jl_value_t*)trilinoswrap::g_rcp_type, jl_svec1(wrapped_t));
      set_julia_type<Teuchos::RCP<T>>(dt);
      protect_from_gc(dt);

      m.method("convert", [] (cxx_wrap::SingletonType<nonconst_t>, const Teuchos::RCP<T>& rcp) { return rcp.get(); });
    }
    return 0;
  }
};

template<typename T>
struct ConvertToJulia<Teuchos::RCP<T>, false, false, false>
{
  inline jl_value_t* operator()(const Teuchos::RCP<T>& cpp_obj) const
  {
    return create<Teuchos::RCP<T>>(cpp_obj);
  }
};

}

#endif
