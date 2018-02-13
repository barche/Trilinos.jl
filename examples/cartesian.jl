"""
2D Cartesian grid functionality
"""
module Cartesian

export CartesianGrid, globalnodes, origin, numglobalnodes, CartesianCoordinates, coordinates, localnodes, partition

# CartesianRange implementation that implements AbstractArray
struct CartesianRange{N,R<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractArray{CartesianIndex{N},N}
  indices::R
end
CartesianRange(inds::NTuple{N,AbstractUnitRange{Int}}) where {N} = CartesianRange{N,typeof(inds)}(inds)
Base.IndexStyle(::Type{CartesianRange{N,R}}) where {N,R} = IndexCartesian()
@inline Base.getindex(A::CartesianRange{N,R}, I::Vararg{Int, N}) where {N,R} = CartesianIndex(first.(A.indices) .- 1) + CartesianIndex(I)
Base.size(A::CartesianRange) = length.(A.indices)

"""
Represents a 2D cartesian grid with uniform spacing
"""
struct CartesianGrid
  nx::Int # Number of points in the X direction
  ny::Int # Number of points in the Y direction
  h::Float64 # Spacing
end

"""
List of all the nodes in the mesh
"""
globalnodes(g::CartesianGrid) = CartesianRange((1:g.nx, 1:g.ny))

"""
Origin of the mesh
"""
origin(g::CartesianGrid) = (g.h*(g.nx-1)/2, g.h*(g.ny-1)/2)

"""
Number of nodes in the mesh
"""
numglobalnodes(g::CartesianGrid) = g.nx*g.ny

"""
Stores local nodes for a cartesian grid as N ranges, together with their process rank
"""
struct LocalNodeRange{N} <: AbstractArray{Int,1}
  ranges::NTuple{N,UnitRange{Int}}
  partitions::NTuple{N,Int}
end

Base.IndexStyle(::Type{LocalNodeRange{N}}) where N = IndexLinear()
Base.size(r::LocalNodeRange) = (sum(length.(r.ranges)),)

# Value to subtract from the index for each subrange
@generated function _offsets(t::NTuple{N,Int}) where N
  result = quote end
  retexpr = :(return ())
  push!(result.args, :(x0 = 0))
  push!(retexpr.args[1].args, :x0)
  for i in 1:N-1
    varname = Symbol("x$i")
    prevvar = Symbol("x$(i-1)")
    push!(result.args, :($varname = t[$(i)] + $prevvar))
    push!(retexpr.args[1].args, varname)
  end
  push!(result.args, retexpr)
  return result
end
_offsets(r::LocalNodeRange{N}) where N = _offsets(length.(r.ranges))

# Index of the range and its offset for the given index
@inline function _rangeindex(r::LocalNodeRange{N}, i::Int) where N
  @boundscheck checkbounds(r, i)
  offsets = _offsets(r)
  for rng_i in 1:N-1
    if offsets[rng_i+1] >= i
      return (rng_i,offsets[rng_i])
    end
  end
  return (N,offsets[N])
end

@inline function Base.getindex(r::LocalNodeRange, i::Int)
  (rng_i,offset) = _rangeindex(r,i)
  return r.ranges[rng_i][i-offset]
end

@inline function partition(r::LocalNodeRange, i::Int)
  (rng_i,offset) = _rangeindex(r,i)
  return r.partitions[rng_i]
end

"""
List of the local nodes in the mesh, using the total number of processes and a 1-based process ID
"""
function localnodes(g::CartesianGrid, numprocs, myid)
  g_n_nodes = numglobalnodes(g)
  my_n_nodes = g_n_nodes ÷ numprocs
  remainder = g_n_nodes % numprocs

  range_start = 1
  for i in 1:(myid-1)
    range_start += my_n_nodes + (i <= remainder ? 1 : 0)
  end

  range_end = range_start + my_n_nodes - (myid <= remainder ? 0 : 1)

  partitions = (myid,myid-1,myid+1)
  gnodes = globalnodes(g)
  pre_start = myid != 1 ? range_start-g.nx : range_start # Start of ghosts before own nodes
  post_end =  myid != numprocs ? range_end+g.nx : range_end # Start of ghosts after own nodes

  # Store the ghosts after the local nodes
  return LocalNodeRange((range_start:range_end,pre_start:(range_start-1),(range_end+1):post_end), partitions)
end

struct CartesianCoordinates{T<:Number} <: AbstractArray{NTuple{2,T},2}
  grid::CartesianGrid
  origin::NTuple{2,T}
  CartesianCoordinates{T}(g::CartesianGrid) where T = new(g, origin(g))
end

CartesianCoordinates(g::CartesianGrid) = CartesianCoordinates{Float64}(g)

Base.IndexStyle(::Type{CartesianCoordinates{T}}) where {T} = IndexCartesian()
@inline function Base.getindex(coords::CartesianCoordinates{T}, I::Vararg{Int, N}) where {T,N}
  return ((I[1]-1)*coords.grid.h, (I[2]-1)*coords.grid.h) .- coords.origin
end
Base.size(coords::CartesianCoordinates) = (coords.grid.nx, coords.grid.ny)

struct LinearCoordinates{T<:Number} <: AbstractArray{T,2}
  grid::CartesianGrid
  origin::NTuple{2,T}
  LinearCoordinates{T}(g::CartesianGrid) where T = new(g, origin(g))
end

coordinates(g::CartesianGrid) = LinearCoordinates{Float64}(g)

Base.IndexStyle(::Type{LinearCoordinates{T}}) where {T} = IndexCartesian()
@inline function Base.getindex(coords::LinearCoordinates{T}, i::Int, j::Int) where {T}
  return (globalnodes(coords.grid)[i][j]-1)*coords.grid.h - coords.origin[j]
end
Base.size(coords::LinearCoordinates) = (numglobalnodes(coords.grid), 2)

end
