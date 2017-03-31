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
template<typename T> struct IsSmartPointerType<Teuchos::RCP<T>> : std::true_type { };
template<typename T> struct IsSmartPointerType<Teuchos::Ptr<T>> : std::true_type { };
template<typename T> struct ConstructorPointerType<Teuchos::Ptr<T>> { typedef Teuchos::RCP<T> type; };

template<typename T>
struct ConstructFromOther<Teuchos::Ptr<T>, Teuchos::RCP<T>>
{
  static jl_value_t* apply(jl_value_t* smart_void_ptr)
  {
    if(jl_typeof(smart_void_ptr) != (jl_value_t*)julia_type<Teuchos::RCP<T>>())
    {
      jl_error("Invalid smart pointer convert, Teuchos::Ptr must be converted from Teuchos::RCP");
      return nullptr;
    }
    auto smart_ptr = unbox_wrapped_ptr<Teuchos::RCP<T>>(smart_void_ptr);
    return boxed_cpp_pointer(new Teuchos::Ptr<T>(smart_ptr->ptr()), static_type_mapping<Teuchos::Ptr<T>>::julia_type(), true);
  }
};

// ArrayView
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
