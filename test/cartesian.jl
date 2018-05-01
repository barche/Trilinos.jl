using Base.Test
using BenchmarkTools

include(joinpath(dirname(dirname(@__FILE__)), "examples", "cartesian.jl"))

using Cartesian

const nx = 10
const h = 2/(nx-1)
const cgrid = CartesianGrid(nx,nx,h)

@test nb_nodes(cgrid) == nx*nx
const crds = coordinates(cgrid)
@test crds[1,1:2] == [-1.0, -1.0]
@test crds[end,1:2] == [1.0, 1.0]

@btime nb_neighbors(cgrid, CartesianIndex(2,2), (g,I) -> I[1] == 1)

# if isinteractive()
#   using Plots
#   scatter(crds[:,1], crds[:,2])
# end