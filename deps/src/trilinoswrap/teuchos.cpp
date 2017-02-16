#include <cxx_wrap.hpp>
#include <mpi.h>

#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "teuchos.hpp"

namespace cxx_wrap
{
  // Support MPI.CComm directly
  template<> struct static_type_mapping<MPI_Comm>
  {
    typedef MPI_Comm type;
    static jl_datatype_t* julia_type() { return ::cxx_wrap::julia_type("CComm", "MPI"); }
    template<typename T> using remove_const_ref = cxx_wrap::remove_const_ref<T>;
  };

  template<typename T> struct DefaultConstructible<Teuchos::Comm<T>> : std::false_type {};
  template<> struct DefaultConstructible<Teuchos::ParameterList> : std::false_type {};

  template<typename T> struct CopyConstructible<Teuchos::Comm<T>> : std::false_type {};
  template<> struct CopyConstructible<Teuchos::ParameterList> : std::false_type {};
}

namespace trilinoswrap
{

jl_datatype_t* g_rcp_type;
jl_datatype_t* g_ptr_type;

jl_datatype_t* rcp_wrappable()
{
  return cxx_wrap::julia_type("RCPWrappable", "Trilinos");
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

template<typename T>
struct AddSetMethod
{
  void operator()(cxx_wrap::Module& mod)
  {
    mod.method("set", [] (Teuchos::ParameterList& pl, const std::string& name, const T& value) { pl.set(name, value); });
  }
};

template<typename T>
struct AddSetMethod<cxx_wrap::StrictlyTypedNumber<T>>
{
  void operator()(cxx_wrap::Module& mod)
  {
    mod.method("set", [] (Teuchos::ParameterList& pl, const std::string& name, const cxx_wrap::StrictlyTypedNumber<T> value) { pl.set(name, value.value); });
  }
};

template<int Dummy=0> void wrap_set(cxx_wrap::Module&) {}

template<typename T, typename... TypesT>
void wrap_set(cxx_wrap::Module& mod)
{
  AddSetMethod<T>()(mod);
  wrap_set<TypesT...>(mod);
}

template<int Dummy=0> void wrap_get(cxx_wrap::Module&) {}

template<typename T, typename... TypesT>
void wrap_get(cxx_wrap::Module& mod)
{
  mod.method("get", [] (cxx_wrap::SingletonType<T>, Teuchos::ParameterList& pl, const std::string& name) -> T { return pl.get<T>(name); });
  wrap_get<TypesT...>(mod);
}

template<int Dummy=0> void wrap_is_type(cxx_wrap::Module&) {}

template<typename T, typename... TypesT>
void wrap_is_type(cxx_wrap::Module& mod)
{
  mod.method("isType", [] (cxx_wrap::SingletonType<T>, Teuchos::ParameterList& pl, const std::string& name) { return pl.isType<T>(name); });
  wrap_is_type<TypesT...>(mod);
}

template<int Dummy=0> jl_datatype_t* get_type(Teuchos::ParameterList& pl, const std::string& pname)
{
  std::cout << "warning, unknown type for parameter " << pname << " of parameterlist " << pl.name() << std::endl;
  return cxx_wrap::static_type_mapping<void>::julia_type();
}

template<typename T, typename... TypesT>
jl_datatype_t* get_type(Teuchos::ParameterList& pl, const std::string& pname)
{
  if(pl.isType<T>(pname))
  {
    return cxx_wrap::static_type_mapping<T>::julia_type();
  }
  return get_type<TypesT...>(pl, pname);
}

void register_teuchos(cxx_wrap::Module& mod)
{
  using namespace cxx_wrap;

  // RCP
  mod.add_type<Parametric<TypeVar<1>>>("RCP");
  mod.add_type<Parametric<TypeVar<1>>>("RCPPtr");
  g_rcp_type = mod.get_julia_type("RCP");
  g_ptr_type = mod.get_julia_type("RCPPtr");

  // Comm
  mod.add_abstract<Teuchos::Comm<int>>("Comm", rcp_wrappable())
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

  mod.add_bits<Teuchos::EVerbosityLevel>("EVerbosityLevel");
  mod.set_const("VERB_DEFAULT", Teuchos::VERB_DEFAULT);
  mod.set_const("VERB_NONE", Teuchos::VERB_NONE);
  mod.set_const("VERB_LOW", Teuchos::VERB_LOW);
  mod.set_const("VERB_MEDIUM", Teuchos::VERB_MEDIUM);
  mod.set_const("VERB_HIGH", Teuchos::VERB_HIGH);
  mod.set_const("VERB_EXTREME", Teuchos::VERB_EXTREME);

  mod.add_type<Teuchos::ParameterList>("ParameterList", cxx_wrap::julia_type("RCPAssociative", "Trilinos"))
    .method("numParams", &Teuchos::ParameterList::numParams)
    .method("setName", &Teuchos::ParameterList::setName)
    .method("name", static_cast<const std::string& (Teuchos::ParameterList::*)() const>(&Teuchos::ParameterList::name))
    .method("print", static_cast<void (Teuchos::ParameterList::*)() const>(&Teuchos::ParameterList::print))
    .method("isParameter", &Teuchos::ParameterList::isParameter)
    .method("isSublist", &Teuchos::ParameterList::isSublist);
  mod.method("ParameterList", [] () { return Teuchos::rcp(new Teuchos::ParameterList()); });
  mod.method("ParameterList", [] (const std::string& name) { return Teuchos::rcp(new Teuchos::ParameterList(name)); });
  mod.method("writeParameterListToXmlFile", [] (const Teuchos::ParameterList& pl, const std::string& filename) { Teuchos::writeParameterListToXmlFile(pl, filename); });
  wrap_set<StrictlyTypedNumber<int32_t>, StrictlyTypedNumber<int64_t>, StrictlyTypedNumber<double>, std::string, bool>(mod);
  wrap_get<int32_t, int64_t, double, std::string, bool>(mod);
  wrap_is_type<int32_t, int64_t, double, std::string, bool>(mod);
  mod.method("get_type", get_type<int32_t, int64_t, double, std::string, bool>);
  mod.method("sublist", [] (Teuchos::ParameterList& pl, const std::string& name) -> Teuchos::ParameterList& { return pl.sublist(name); });
  mod.method("keys", [] (const Teuchos::ParameterList& pl)
  {
    cxx_wrap::Array<std::string> keys;
    JL_GC_PUSH1(keys.gc_pointer());
    for(const auto& elem : pl)
    {
      keys.push_back(elem.first);
    }
    JL_GC_POP();
    return (jl_value_t*)keys.wrapped();
  });
}

} // namespace trilinoswrap
