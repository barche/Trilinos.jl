using Test
using BenchmarkTools

include(joinpath(dirname(dirname(@__FILE__)), "examples", "cartesian.jl"))

using .Cartesian

const nx = 10
const h = 2/(nx-1)
println("building grid")
const cgrid = CartesianGrid(nx,nx,h)

println("tests...")
@test nb_nodes(cgrid) == nx*nx
const crds = coordinates(cgrid)
@test crds[1,1:2] == [-1.0, -1.0]
@test crds[end,1:2] == [1.0, 1.0]

println("nb neighbors...")
@btime nb_neighbors(cgrid, 4, (g,I) -> I[1] == 1)

# if isinteractive()
#   using Plots
#   scatter(crds[:,1], crds[:,2])
# end