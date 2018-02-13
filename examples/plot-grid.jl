using MPI
using Trilinos
using Base.Test
using Compat
using BenchmarkTools
#using Cartesian

if isinteractive()
  using Plots
end

const IndexT = Int


# abstract type IndexMapping end

# """
# Store a global mapping between grid GID and linear system GIDs
# """
# struct FilteredMapping{ArrayT} <: IndexMapping
#   array::ArrayT
#   grid::CartesianGrid
# end

# function FilteredMapping(grid, nodes::Function, filter::Function, startindex)
#   mappingarray = fill(-1, size(nodes(grid)))
#   gid = startindex
#   for I in nodes(grid)
#     if filter(grid, I)
#       mappingarray[I] = gid
#       gid += 1
#     end
#   end
#   return FilteredMapping(mappingarray, grid)
# end

# Base.getindex(m::IndexMapping, I::CartesianIndex) = m.array[I]
# Base.in(I::CartesianIndex, m::IndexMapping) = m[I] != -1

# function test_grid_functions(myid, numprocs)
#   nx = 5
#   h = 2/(nx-1)
#   grid = CartesianGrid(nx,nx,h)

#   @test length(globalnodes(grid)) == nx*nx
#   @test numglobalnodes(grid) == nx*nx

#   cartesian_coords = CartesianCoordinates(grid)

#   for I in globalnodes(grid)
#     @test cartesian_coords[I] == ((I[1]-1)*h-1.0, (I[2]-1)*h-1.0)
#   end

#   coords = coordinates(grid)

#   for (i,I) in enumerate(globalnodes(grid))
#     @test cartesian_coords[I] == (coords[i,1], coords[i,2])
#   end

#   # mesh_to_system_gid = FilteredMapping(grid, globalnodes, (::Any,::Any) -> true, 0)
  
#   # @show localnodes(grid, 4, 1)
#   # @show localnodes(grid, 4, 2)
#   # @show localnodes(grid, 4, 3)
#   # @show localnodes(grid, 4, 4)

#   # @show c = coordinates.(grid, globalnodes(grid))
#   # @show reinterpret(Array{Float64,2}, c)
# end

# MPI setup
if !MPI.Initialized()
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
my_rank = Teuchos.getRank(comm)
nb_procs = Teuchos.getSize(comm)

if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

# @testset "Grid tests" begin
#   test_grid_functions(my_rank+1, nb_procs)
# end

if isinteractive()
  
  const nx = 10
  const h = 2/(nx-1)
  const grid = CartesianGrid(nx,nx,h)

  coords = coordinates(grid)

  nb_procs = 4
  @show ln1 = localnodes(grid, nb_procs, 1)
  @show ln2 = localnodes(grid, nb_procs, 2)
  @show ln3 = localnodes(grid, nb_procs, 3)
  @show ln4 = localnodes(grid, nb_procs, 4)

  o1 = filter(i -> (partition(ln1,i) == 1), eachindex(ln1))
  o2 = filter(i -> (partition(ln2,i) == 2), eachindex(ln2))
  o3 = filter(i -> (partition(ln3,i) == 3), eachindex(ln3))
  o4 = filter(i -> (partition(ln4,i) == 4), eachindex(ln4))

  g1 = filter(i -> (partition(ln1,i) != 1), eachindex(ln1))
  g2 = filter(i -> (partition(ln2,i) != 2), eachindex(ln2))
  g3 = filter(i -> (partition(ln3,i) != 3), eachindex(ln3))
  g4 = filter(i -> (partition(ln4,i) != 4), eachindex(ln4))

  @show length(ln1)
  @show length(ln2)
  @show length(ln3)
  @show length(ln4)

  @show numglobalnodes(grid)
  @show length(ln1) + length(ln2) + length(ln3) + length(ln4)

  scatter( coords[ln1[o1],1], coords[ln1[o1],2], marker=:utriangle, legend=false, aspectratio=1)
  scatter!(coords[ln2[o2],1], coords[ln2[o2],2], marker=:dtriangle)
  scatter!(coords[ln3[o3],1], coords[ln3[o3],2], marker=:+)
  scatter!(coords[ln4[o4],1], coords[ln4[o4],2], marker=:x)
  scatter!(coords[ln1[g1],1], coords[ln1[g1],2], alpha=0.4, marker=:utriangle, legend=false)
  scatter!(coords[ln2[g2],1], coords[ln2[g2],2], alpha=0.4, marker=:dtriangle)
  scatter!(coords[ln3[g3],1], coords[ln3[g3],2], alpha=0.4, marker=:+)
  scatter!(coords[ln4[g4],1], coords[ln4[g4],2], alpha=0.4, marker=:x)

else
  MPI.Finalize()
end