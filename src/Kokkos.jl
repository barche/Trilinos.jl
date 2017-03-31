module Kokkos
  using CxxWrap, MPI
  import .._l_trilinos_wrap

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)

  """
  Wrap a concrete view type, providing an array interface
  """
  immutable View{ScalarT,N,ViewT} <: AbstractArray{ScalarT,N}
    kokkos_view::ViewT
  end
  View{ScalarT,N,ViewT}(::Type{ScalarT}, ::Type{Val{N}}, v::ViewT) = View{ScalarT,N,ViewT}(v)

  # Store a raw pointer together with array size
  immutable PtrWrapper{ScalarT,N}
    ptr::Ptr{ScalarT}
    size::NTuple{N,Int}
  end

  # Generic array interface, using operator() access
  Base.size{ScalarT, ViewT <: CxxWrap.CppAny}(v::View{ScalarT,1,ViewT}) = (Int(dimension(v.kokkos_view,0)),)
  Base.size{ScalarT, ViewT <: CxxWrap.CppAny}(v::View{ScalarT,2,ViewT}) = (Int(dimension(v.kokkos_view,0)),Int(dimension(v.kokkos_view,1)))
  Base.getindex{ScalarT, ViewT <: CxxWrap.CppAny}(v::View{ScalarT,1,ViewT}, i::Integer) = unsafe_load(v.kokkos_view(i-1, 0))
  Base.getindex{ScalarT, ViewT <: CxxWrap.CppAny}(v::View{ScalarT,2,ViewT}, i::Integer, j::Integer) = unsafe_load(v.kokkos_view(i-1, j-1))
  Base.setindex!{ScalarT, ViewT <: CxxWrap.CppAny}(v::View{ScalarT,1,ViewT}, value, i::Integer) = unsafe_store!(v.kokkos_view(i-1, 0), value)
  Base.setindex!{ScalarT, ViewT <: CxxWrap.CppAny}(v::View{ScalarT,2,ViewT}, value, i::Integer, j::Integer) = unsafe_store!(v.kokkos_view(i-1, j-1), value)


  # raw pointer access (i.e. the fast path)
  Base.linearindexing{ScalarT,N}(::Type{View{ScalarT,N,PtrWrapper{ScalarT,N}}}) = Base.LinearFast()
  Base.size{ScalarT,N}(v::View{ScalarT,N,PtrWrapper{ScalarT,N}}) = v.kokkos_view.size
  Base.getindex{ScalarT,N}(v::View{ScalarT,N,PtrWrapper{ScalarT,N}}, i::Integer) = unsafe_load(v.kokkos_view.ptr,i)
  Base.setindex!{ScalarT,N}(v::View{ScalarT,N,PtrWrapper{ScalarT,N}}, value, i::Integer) = unsafe_store!(v.kokkos_view.ptr,value,i)
end
