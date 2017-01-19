#ifndef TRILINOS_JL_RCP_HPP
#define TRILINOS_JL_RCP_HPP

#include <array.hpp>
#include <cxx_wrap.hpp>

#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_RCP.hpp>

namespace trilinoswrap
{
  extern jl_datatype_t* g_rcp_type;

  jl_datatype_t* julia_rcp_wrappable();
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
    typedef typename std::remove_const<T>::type nonconst_t;
    if(!static_type_mapping<Teuchos::RCP<nonconst_t>>::has_julia_type())
    {
      jl_datatype_t* wrapped_t = static_type_mapping<nonconst_t>::julia_type();

      jl_datatype_t* dt = (jl_datatype_t*)jl_apply_type((jl_value_t*)trilinoswrap::g_rcp_type, jl_svec1(wrapped_t));
      protect_from_gc(dt);
      set_julia_type<Teuchos::RCP<const nonconst_t>>(dt);
      set_julia_type<Teuchos::RCP<nonconst_t>>(dt);
      m.method("convert", [] (cxx_wrap::SingletonType<nonconst_t>, const Teuchos::RCP<nonconst_t>& rcp) { return rcp.get(); });
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

// Transparent conversion from Julia arrays to ArrayView
template<typename T> struct IsValueType<Teuchos::ArrayView<const T>> : std::true_type {};

template<typename T>
struct InstantiateParametricType<Teuchos::ArrayView<const T>>
{
  inline int operator()(Module& m) const
  {
    if(!static_type_mapping<Teuchos::ArrayView<const T>>::has_julia_type())
    {
      jl_datatype_t* dt = (jl_datatype_t*)jl_apply_array_type(static_type_mapping<T>::julia_type(), 1);
      protect_from_gc(dt);
      set_julia_type<Teuchos::ArrayView<const T>>(dt);
    }
    return 0;
  }
};

template<typename T>
struct ConvertToCpp<Teuchos::ArrayView<const T>, false, false, false>
{
  Teuchos::ArrayView<const T> operator()(jl_value_t* val) const
  {
    cxx_wrap::ArrayRef<T,1> arr_ref((jl_array_t*)val);
    return Teuchos::ArrayView<const T>(arr_ref.data(), arr_ref.size());
  }
};

}

#endif
