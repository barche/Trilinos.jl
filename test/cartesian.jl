using Base.Test
using BenchmarkTools

include(joinpath(dirname(dirname(@__FILE__)), "examples", "cartesian.jl"))

using Cartesian

lr = Cartesian.LocalNodeRange((3:7, 1:2, 8:10), (1,2,3))
@test length(lr) == 10
@test size(lr) == (10,)

@test Cartesian._offsets(lr) == (0,5,7)
@test Cartesian._rangeindex(lr,8) == (3,7)
@test Cartesian._rangeindex(lr,9) == (3,7)
@test Cartesian._rangeindex(lr,10) == (3,7)

@test lr[2] == 4
@test lr[5] == 7
@test lr[6] == 1
@test lr[7] == 2
@test lr[8] == 8
@test lr[9] == 9
@test lr[10] == 10
@test_throws BoundsError lr[11]

@test partition(lr,2) == 1
@test partition(lr,5) == 1
@test partition(lr,6) == 2
@test partition(lr,10) == 3
@test_throws BoundsError partition(lr,11)

lr2 = Cartesian.LocalNodeRange((1:0, 1:5, 5:4), (0,1,0))
@test length(lr2) == 5
@test lr2 == collect(1:5)

const nx = 10
const h = 2/(nx-1)
const grid = CartesianGrid(nx,nx,h)

const nb_procs = 4
ln1 = localnodes(grid, nb_procs, 1)
ln2 = localnodes(grid, nb_procs, 2)
ln3 = localnodes(grid, nb_procs, 3)
ln4 = localnodes(grid, nb_procs, 4)

@test length(ln1) == 35
@test length(ln2) == 45
@test length(ln3) == 45
@test length(ln4) == 35
