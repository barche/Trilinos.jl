module Kokkos
  using CxxWrap
  using MPI
  using Compat
  import .._l_trilinos_wrap

  using CustomUnitRanges: filename_for_zerorange
  include(filename_for_zerorange)

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)

  """
  Expose a view using its raw data pointer
  """
  immutable PtrView{ScalarT,N,LayoutT} <: AbstractArray{ScalarT,N}
    ptr::Ptr{ScalarT}
    size::NTuple{N,Int}
  end

  make_dimensions{N}(v,::Type{Val{N}}) = ([Int(Kokkos.dimension(v,i-1)) for i in 1:N]...)

  function view{ST,LayoutT,SpaceT,N}(v::View3{Ptr{Ptr{ST}},LayoutT,SpaceT}, ndims::Type{Val{N}})
    return PtrView{ST,N,LayoutT}(Kokkos.ptr_on_device(v),make_dimensions(v,ndims))
  end

  function view{ST,LayoutT,SpaceT,N}(v::View4{Ptr{Ptr{ST}},LayoutT,SpaceT,Void}, ndims::Type{Val{N}})
    return PtrView{ST,N,LayoutT}(Kokkos.ptr_on_device(v),make_dimensions(v,ndims))
  end

  # Array interface, 0-based for consistency with global-to-local index mapping
  @compat Base.IndexStyle{ScalarT,N,LayoutT}(::Type{PtrView{ScalarT,N,LayoutT}}) = IndexLinear()
  Base.indices{ScalarT,N}(v::PtrView{ScalarT,N,LayoutLeft}) = map(ZeroRange, v.size)
  Base.indices{ScalarT,N}(v::PtrView{ScalarT,N,LayoutLeft},d) = ZeroRange(v.size[d])
  Base.getindex{ScalarT}(v::PtrView{ScalarT,1,LayoutLeft}, i::Integer) = unsafe_load(v.ptr,i+1)
  Base.setindex!{ScalarT}(v::PtrView{ScalarT,1,LayoutLeft}, value, i::Integer) = unsafe_store!(v.ptr,value,i+1)
  Base.getindex{ScalarT,N}(v::PtrView{ScalarT,N,LayoutLeft}, i::Integer) = unsafe_load(v.ptr,i)
  Base.setindex!{ScalarT,N}(v::PtrView{ScalarT,N,LayoutLeft}, value, i::Integer) = unsafe_store!(v.ptr,value,i)
end
