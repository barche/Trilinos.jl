module Kokkos
  using CxxWrap
  using MPI
  import ..libjltrilinos

  using CustomUnitRanges: filename_for_zerorange
  include(filename_for_zerorange)

  @wrapmodule(libjltrilinos, :register_kokkos)

  """
  Expose a view using its raw data pointer
  """
  struct PtrView{ScalarT,N,LayoutT} <: AbstractArray{ScalarT,N}
    ptr::Ptr{ScalarT}
    size::NTuple{N,Int}
    stored_view # prevent GC of the actual view
  end

  make_dimensions(v,::Type{Val{N}}) where {N} = ([Int(Kokkos.dimension(v,i-1)) for i in 1:N]...,)

  function view(v::View3{Ptr{Ptr{ST}},LayoutT,SpaceT}, ndims::Type{Val{N}}) where {ST,LayoutT,SpaceT,N}
    return PtrView{ST,N,LayoutT}(Kokkos.ptr_on_device(v),make_dimensions(v,ndims), v)
  end

  function view(v::View4{Ptr{Ptr{ST}},LayoutT,SpaceT,Nothing}, ndims::Type{Val{N}}) where {ST,LayoutT,SpaceT,N}
    return PtrView{ST,N,LayoutT}(Kokkos.ptr_on_device(v),make_dimensions(v,ndims), v)
  end

  # Array interface, 0-based for consistency with global-to-local index mapping
  Base.IndexStyle(::Type{PtrView{ScalarT,N,LayoutT}}) where {ScalarT,N,LayoutT} = IndexLinear()
  Base.axes(v::PtrView{ScalarT,N,LayoutLeft}) where {ScalarT,N} = map(ZeroRange, v.size)
  Base.axes(v::PtrView{ScalarT,N,LayoutLeft},d) where {ScalarT,N} = ZeroRange(v.size[d])
  Base.getindex(v::PtrView{ScalarT,1,LayoutLeft}, i::Integer) where {ScalarT} = unsafe_load(v.ptr,i+1)
  Base.setindex!(v::PtrView{ScalarT,1,LayoutLeft}, value, i::Integer) where {ScalarT} = unsafe_store!(v.ptr,value,i+1)
  Base.getindex(v::PtrView{ScalarT,N,LayoutLeft}, i::Integer) where {ScalarT,N} = unsafe_load(v.ptr,i)
  Base.setindex!(v::PtrView{ScalarT,N,LayoutLeft}, value, i::Integer) where {ScalarT,N} = unsafe_store!(v.ptr,value,i)
  Base.similar(v::PtrView{ScalarT,N,LayoutT}) where {ScalarT,N,LayoutT} = Array{ScalarT,N}(undef, length.(axes(v)))
  Base.size(v::PtrView) = v.size

  function Base.copy!(dest::AbstractArray{T,N}, src::PtrView{T,N,LayoutT}) where {T,N,LayoutT}
    println("copying")
    @show maximum(src)
    @boundscheck size.(axes(src)) == size.(axes(dest))
    for (I,J) in zip(eachindex(IndexStyle(dest), dest), eachindex(IndexStyle(src), src))
      @inbounds dest[I] = src[J]
    end
    return dest
  end

  struct StaticCrsGraph
    rowmap::Vector{Int}
    entries::Vector{Int}
  
    function StaticCrsGraph(nb_rows, nb_entries)
      rowmap = Vector{Int}(undef, nb_rows+1)
      entries = Vector{Int}(undef, nb_entries)
      rowmap[end] = nb_entries
      return new(rowmap,entries)
    end
  end

  get_datatype(::Type{View3{Ptr{DT},LayoutT,SpaceT}}) where {DT,LayoutT,SpaceT} = DT
  get_datatype(::Type{View3{CxxWrap.ConstPtr{DT},LayoutT,SpaceT}}) where {DT,LayoutT,SpaceT} = DT

  function to_cpp(g::StaticCrsGraph, ::Type{GraphT}) where {GraphT}
    entries_t = entries_type(GraphT)
    rowmap_t = row_map_type(GraphT)
    return GraphT(makeview(entries_t, "entries", convert(Vector{get_datatype(entries_t)}, g.entries)), makeview(rowmap_t, "rowmap", convert(Vector{get_datatype(rowmap_t)}, g.rowmap)))
  end
end
