"""
2D Cartesian grid functionality
"""
module Cartesian

using Compat

export CartesianGrid, nb_nodes, origin, coordinates, cartesianindices, nb_neighbors, setneighbors!, dirichletrhs

"""
Represents a 2D cartesian grid with uniform spacing
"""
struct CartesianGrid
  nx::Int # Number of points in the X direction
  ny::Int # Number of points in the Y direction
  h::Float64 # Spacing
end

cartesianindices(g::CartesianGrid) = CartesianIndices((g.nx, g.ny))
linearindices(g::CartesianGrid) = LinearIndices((g.nx, g.ny))

"""
Origin of the mesh
"""
origin(g::CartesianGrid) = (g.h*(g.nx-1)/2, g.h*(g.ny-1)/2)

"""
Number of nodes in the mesh
"""
nb_nodes(g::CartesianGrid) = g.nx*g.ny

struct LinearCoordinates{T<:Number} <: AbstractArray{T,2}
  grid::CartesianGrid
  origin::NTuple{2,T}
  LinearCoordinates{T}(g::CartesianGrid) where T = new(g, origin(g))
end

coordinates(g::CartesianGrid) = LinearCoordinates{Float64}(g)

Base.IndexStyle(::Type{LinearCoordinates{T}}) where {T} = IndexCartesian()

@inline function Base.getindex(coords::LinearCoordinates{T}, i::Int, j::Int) where {T}
  tocartesian = cartesianindices(coords.grid)
  return (tocartesian[i][j]-1)*coords.grid.h - coords.origin[j]
end

Base.size(coords::LinearCoordinates) = (nb_nodes(coords.grid), 2)

function laplace2D_stencil(I::CartesianIndex)
  return CartesianIndex.((0,0,-1, 0,1),
                         (0,1, 0,-1,0)) .+ I
end

function nb_neighbors(grid::CartesianGrid, gid::Integer, f::Function)
  return count(i -> i ∈ cartesianindices(grid) && f(grid, i), laplace2D_stencil(cartesianindices(grid)[gid]))
end

function setneighbors!(array, startidx, grid::CartesianGrid, gid::Integer, f::Function)
  cartinds = cartesianindices(grid)
  lininds = linearindices(grid)
  i = startidx
  for I in laplace2D_stencil(cartinds[gid])
    if I ∈ cartinds && f(grid,I)
      array[i] = lininds[I]
      i += 1
    end
  end
  return i - startidx
end

function dirichletrhs(grid::CartesianGrid, gid::Integer, isdirichlet::Function, dirichletval::Function)
  cartinds = cartesianindices(grid)
  lininds = linearindices(grid)
  result = 0.0
  for I in laplace2D_stencil(cartinds[gid])
    if I ∈ cartinds && isdirichlet(grid,I)
      result += dirichletval(grid, I)
    end
  end
  return result
end

end
