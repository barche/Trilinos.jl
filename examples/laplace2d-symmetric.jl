using MPI
using Trilinos
#using BenchmarkTools

include("cartesian.jl")
using Cartesian

if isinteractive()
  using Plots
end

const IndexT = Int

"""
Determine if a point is part of the Dirichlet boundary conditions
"""
function isdirichlet(grid, I)
  if(I[1] == 1 || I[1] == grid.nx || I[2] == 1 || I[2] == grid.ny)
    return true
  end
  return false
end

"""
Get the ditichlet boundary condition value for the given point
""" 
function dirichlet(grid, I)
  return 0.0
end

function laplace2d(comm, nx, ny)
  h = 2/(nx-1)
  grid = CartesianGrid(nx,nx,h)
  my_partition = my_rank+1 # Grid partitions are 1-based while MPI ranks are 0-based

  allnodes = globalnodes(grid)
  localgridnodes = localnodes(grid, nb_procs, my_partition)
  localsystemnodes = filter(i -> !isdirichlet(grid, allnodes[localgridnodes[i]]), localgridnodes)

  @show nsysnodes = count(I -> !isdirichlet(grid,I), allnodes)

  display(localsystemnodes)
end

# MPI setup
if !MPI.Initialized()
  MPI.Init()
end

const comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const my_rank = Teuchos.getRank(comm)
const nb_procs = Teuchos.getSize(comm)

if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

const nx = 10
laplace2d(comm, nx, nx)

if isinteractive()

else
  MPI.Finalize()
end