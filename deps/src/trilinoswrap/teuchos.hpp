#ifndef TRILINOS_JL_RCP_HPP
#define TRILINOS_JL_RCP_HPP

#include <cxx_wrap.hpp>
#include <Teuchos_RCP.hpp>

namespace trilinoswrap
{
  extern jl_datatype_t* g_rcp_type;
}

namespace cxx_wrap
{

// Some special-casing for RCP

/// Access to the RCP dt from any module
static jl_datatype_t*& rcp_type()
{
  static jl_datatype_t* m_type_pointer = nullptr;
  return m_type_pointer;
}

template<typename T>
struct InstantiateParametricType<Teuchos::RCP<T>>
{
  inline int operator()(Module&) const
  {
    // Register the Julia type if not already instantiated
    if(!static_type_mapping<Teuchos::RCP<T>>::has_julia_type())
    {
      jl_datatype_t* dt = (jl_datatype_t*)jl_apply_type((jl_value_t*)trilinoswrap::g_rcp_type, jl_svec1(static_type_mapping<typename std::remove_const<T>::type>::julia_type()));
      set_julia_type<Teuchos::RCP<T>>(dt);
      protect_from_gc(dt);
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
