module Kokkos
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..RCPWrappable
  import ..RCPAssociative

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)

  """
  Wrap a concrete view type, providing an array interface
  """
  immutable View{ScalarT,N,ViewT} <: AbstractArray{ScalarT,N}
    kokkos_view::ViewT
  end

  View{ScalarT,N,ViewT}(::Type{ScalarT}, ::Type{Val{N}}, v::ViewT) = View{ScalarT,N,ViewT}(v)

  Base.size{ScalarT, ViewT}(v::View{ScalarT,1,ViewT}) = (Int(dimension(v.kokkos_view,0)),)
  Base.size{ScalarT, ViewT}(v::View{ScalarT,2,ViewT}) = (Int(dimension(v.kokkos_view,0)),Int(dimension(v.kokkos_view,1)))

  Base.getindex{ScalarT, ViewT}(v::View{ScalarT,1,ViewT}, i::Integer) = unsafe_load(v.kokkos_view(i-1, 0))
  Base.getindex{ScalarT, ViewT}(v::View{ScalarT,2,ViewT}, i::Integer, j::Integer) = unsafe_load(v.kokkos_view(i-1, j-1))

  Base.setindex!{ScalarT, ViewT}(v::View{ScalarT,1,ViewT}, value, i::Integer) = unsafe_store!(v.kokkos_view(i-1, 0), value)
  Base.setindex!{ScalarT, ViewT}(v::View{ScalarT,2,ViewT}, value, i::Integer, j::Integer) = unsafe_store!(v.kokkos_view(i-1, j-1), value)
end
