#ifndef TRILINOS_JL_TEUCHOS_HPP
#define TRILINOS_JL_TEUCHOS_HPP

#include <array.hpp>
#include <cxx_wrap.hpp>

#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_VerbosityLevel.hpp>

namespace trilinoswrap
{
  extern jl_datatype_t* g_rcp_type;
  extern jl_datatype_t* g_ptr_type;

  jl_datatype_t* rcp_wrappable();

  /// Helper function to generate RCP conversions between convertible C++ types
  template<typename FromT, typename ToT>
  inline Teuchos::RCP<ToT> convert(cxx_wrap::SingletonType<Teuchos::RCP<ToT>>, const Teuchos::RCP<FromT>& rcp)
  {
    return Teuchos::RCP<ToT>(rcp);
  }

  template<typename FromT, typename ToT>
  inline ToT* convert_unwrap(cxx_wrap::SingletonType<ToT>, const Teuchos::RCP<FromT>& rcp)
  {
    return rcp.get();
  }

  template<typename T>
  struct ArrayViewMirror
  {
    T* array;
    int_t size;
  };
}

namespace cxx_wrap
{

template<> struct IsBits<MPI_Comm> : std::true_type {};
template<> struct IsBits<Teuchos::ETransp> : std::true_type {};
template<> struct IsBits<Teuchos::EVerbosityLevel> : std::true_type {};

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
struct InstantiateParametricType<Teuchos::Ptr<T>>
{
  inline int operator()(Module& m) const
  {
    // Register the Julia type if not already instantiated
    typedef typename std::remove_const<T>::type nonconst_t;
    if(!static_type_mapping<Teuchos::Ptr<nonconst_t>>::has_julia_type())
    {
      jl_datatype_t* wrapped_t = static_type_mapping<nonconst_t>::julia_type();

      jl_datatype_t* dt = (jl_datatype_t*)jl_apply_type((jl_value_t*)trilinoswrap::g_ptr_type, jl_svec1(wrapped_t));
      protect_from_gc(dt);
      set_julia_type<Teuchos::Ptr<const nonconst_t>>(dt);
      set_julia_type<Teuchos::Ptr<nonconst_t>>(dt);
      m.method("convert", [] (cxx_wrap::SingletonType<Teuchos::Ptr<nonconst_t>>, const Teuchos::RCP<nonconst_t>& rcp) { return rcp.ptr(); });
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

template<typename T>
struct ConvertToJulia<Teuchos::Ptr<T>, false, false, false>
{
  inline jl_value_t* operator()(const Teuchos::Ptr<T>& cpp_obj) const
  {
    return create<Teuchos::Ptr<T>>(cpp_obj);
  }
};

template<typename T> struct IsBits<Teuchos::ArrayView<T>> : std::true_type {};
template<typename T> struct IsImmutable<Teuchos::ArrayView<T>> : std::true_type {};

template<typename T>
struct static_type_mapping<Teuchos::ArrayView<T>>
{
  typedef typename std::remove_const<T>::type NonConstT;
  typedef trilinoswrap::ArrayViewMirror<T> type;
  static jl_datatype_t* julia_type() { return (jl_datatype_t*)apply_type((jl_value_t*)cxx_wrap::julia_type("ArrayView", "Teuchos"), jl_svec1(static_type_mapping<NonConstT>::julia_type())); }
};


template<typename T>
struct ConvertToCpp<Teuchos::ArrayView<T>, false, true, true>
{
  Teuchos::ArrayView<T> operator()(trilinoswrap::ArrayViewMirror<T> arr_ref) const
  {
    return Teuchos::ArrayView<T>(arr_ref.array, arr_ref.size);
  }
};

}

#endif
