using MPI
using MPIArrays
using Trilinos
using BenchmarkTools
using Compat
using Base.Test

include("cartesian.jl")
using Cartesian

if isinteractive()
  using Plots
end

const IndexT = Int32

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
Get the dirichlet boundary condition value for the given point
""" 
function dirichlet(grid, I)
  return 0.0
end

function buildgraph(comm, grid)
  mpicomm = MPI.Comm(comm)
  cartinds = cartesianindices(grid)

  # Map system GID to grid GID
  systogrid = MPIArray{Int}(mpicomm, (nb_procs,), length(cartinds))
  forlocalpart!(lp -> lp .= localindices(systogrid)[1], systogrid)
  sync(systogrid)
  filter!(i -> !isdirichlet(grid, cartinds[i]), systogrid)
  redistribute!(systogrid)
  
  # Mapping from grid GID to system GID (or -1 if not part of system)
  gridtosys = MPIArray{Int}(mpicomm, (nb_procs,), length(cartinds))
  forlocalpart!(lp -> fill!(lp, -1), gridtosys)
  sync(gridtosys)

  forlocalpart!(systogrid) do sys_lp
    sysgids = localindices(systogrid)[1]
    gridblock = gridtosys[minimum(sys_lp):maximum(sys_lp)]
    gridarr = getblock(gridblock)
    gridglb = GlobalBlock(gridarr, gridblock)
    for (i,gid) in enumerate(sys_lp)
      gridglb[gid] = sysgids[i]
    end
    putblock!(gridarr, gridblock)
  end

  sync(gridtosys)
  
  # Count the number of non-zero entries in the graph
  my_nnz = forlocalpart(systogrid) do sys_lp
    result = 0
    for gridgid in sys_lp
      result += nb_neighbors(grid, gridgid, !isdirichlet)
    end
    return result
  end

  # Fill the graph with mesh global IDs
  my_nrows = length(localindices(systogrid)[1])
  staticgraph = Kokkos.StaticCrsGraph(my_nrows,my_nnz)
  forlocalpart(systogrid) do sys_lp
    counter = 0
    for (i,gridgid) in enumerate(sys_lp)
      staticgraph.rowmap[i] = counter
      counter += setneighbors!(staticgraph.entries, staticgraph.rowmap[i]+1, grid, gridgid, !isdirichlet)
    end
    @assert counter == staticgraph.rowmap[end]
  end

  # Sync missing off-processor gridnodes
  gridghosts = GhostedBlock(gridtosys, staticgraph.entries)
  
  # Convert the graph from mesh global IDs to system global IDs
  map!(x -> getglobal(gridghosts,x), staticgraph.entries, staticgraph.entries)

  # sync missing off-processor linear system entries
  sysghosts = GhostedBlock(systogrid, staticgraph.entries)

  # Convert the graph from system global IDs to system local IDs
  map!(x -> globaltolocal(sysghosts,x)-1, staticgraph.entries, staticgraph.entries)
  
  # Construct the Tpetra maps
  rowmap = Tpetra.Map(length(systogrid), Teuchos.ArrayView(collect(localindices(systogrid)...)), 1, comm)
  col_glb_ids = globalids(sysghosts)
  nb_glb_col_ids = MPI.Allreduce(length(col_glb_ids), +, mpicomm)
  colmap = Tpetra.Map(nb_glb_col_ids, Teuchos.ArrayView(col_glb_ids), 1, comm)

  graph = Tpetra.CrsGraph(rowmap, colmap, staticgraph)
  @assert Tpetra.isFillComplete(graph)

  return graph, sysghosts
end

function fillmatrix!(A, grid)
  nb_rows = Int(Tpetra.getNodeNumRows(A))
  @assert Tpetra.isLocallyIndexed(A)

  row_indices = zeros(IndexT,5)
  row_values_dummy = zeros(Float64,5)
  row_values = fill(-1.0, 5)
  row_values[1] = 4.0

  indices_view = Teuchos.ArrayView(row_indices)
  values_view = Teuchos.ArrayView(row_values_dummy)
  rowlength_ref = Ref(UInt(0))

  Tpetra.resumeFill(A)
  for row in 1:nb_rows
    Tpetra.getLocalRowCopy(A, row-1, indices_view, values_view, rowlength_ref)
    rowlength = Int(rowlength_ref[])
    Tpetra.replaceLocalValues(A, row-1, Teuchos.ArrayView(row_indices,rowlength), Teuchos.ArrayView(row_values,rowlength))
  end
  Tpetra.fillComplete(A)
end

function fillrhs!(b, grid, systogrid)
  b_view = Tpetra.device_view(b)
  gcoords = coordinates(grid)
  for i in linearindices(b_view)
    gid = systogrid[i+1]
    x = gcoords[gid,1]
    y = gcoords[gid,2]
    b_view[i] = 2*grid.h^2*((1-x^2)+(1-y^2)) + dirichletrhs(grid, gid, isdirichlet, dirichlet)
  end
end

function laplace2d(comm, grid)
  println("Graph time:")
  @time (graph, systogrid) = buildgraph(comm, grid)

  A = Tpetra.CrsMatrix(graph)
  println("Matrix fill time:")
  @time fillmatrix!(A, grid)
  
  rangemap = Tpetra.getRangeMap(graph)
  b = Tpetra.Vector(rangemap)
  println("RHS fill time:")
  @time fillrhs!(b, grid, systogrid)

  maxiter = 1000

  params = Trilinos.default_parameters("CG")
  solver_params = params["Linear Solver Types"]["Belos"]["Solver Types"]["CG"]
  solver_params["Convergence Tolerance"] = 1e-12
  solver_params["Verbosity"] = Belos.StatusTestDetails + Belos.FinalSummary + Belos.TimingDetails
  solver_params["Maximum Iterations"] = Int32(maxiter)
  params["Preconditioner Type"] = "MueLu"
  
  # params = Trilinos.default_parameters()
  # solver_params = params["Linear Solver Types"]["Belos"]["Solver Types"]["BLOCK GMRES"]
  # solver_params["Convergence Tolerance"] = 1e-12
  # solver_params["Verbosity"] = Belos.StatusTestDetails + Belos.FinalSummary + Belos.TimingDetails
  # solver_params["Num Blocks"] = Int32(maxiter)
  # solver_params["Maximum Iterations"] = Int32(maxiter)
  # params["Preconditioner Type"] = "Ifpack2"

  solver = TpetraSolver(A, params)

  return (solver,b,systogrid)
end

# MPI setup
if !MPI.Initialized()
  MPI.Init()
  MPI.finalize_atexit()
end

const comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const my_rank = Teuchos.getRank(comm)
const nb_procs = Int(Teuchos.getSize(comm))

if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

function check_solution(sol, g::CartesianGrid)
  coords = coordinates(g)
  gids = LinearIndices(sol)
  cartinds = CartesianIndices(sol)[localindices(sol)...]
  
  return forlocalpart(sol) do lp
    errorcount = 0
    for (φ,I) in zip(lp,cartinds)
      gid = gids[I]
      x = coords[gid,1]
      y = coords[gid,2]
      errorcount +=  Int(abs(φ - (1-x^2)*(1-y^2)) > 1e-10)
    end
    return errorcount
  end
end

const nx = 1001

function solve_laplace2d(comm)
  h = 2/(nx-1)
  grid = CartesianGrid(nx,nx,h)

  (lows,b,systogrid) = laplace2d(comm, grid)
  (lows,b,systogrid) = laplace2d(comm, grid)
  (lows,b,systogrid) = laplace2d(comm, grid)

  # Compute the solution
  solvec = lows \ b

  cartinds = cartesianindices(grid)
  # Solution in grid coordinates
  gridsol = MPIArray{Float64}(MPI.Comm(comm), (nb_procs,1), grid.nx, grid.ny)
  
  myrange = CartesianIndices(localindices(gridsol))
  
  solview = Tpetra.device_view(solvec)
  ghosts = Dict{Int,CartesianIndex{2}}()
  forlocalpart!(gridsol) do lp
    myblock = GlobalBlock(lp, gridsol[localindices(gridsol)...])
    for (i,x) in enumerate(solview)
      gid = systogrid[i]
      I = cartinds[gid]
      if I ∈ myrange
        myblock[I] = x
      else
        ghosts[i-1] = I
      end
    end
  end

  for (i,I) in ghosts
    gridsol[I] = solview[i]
  end
  
  for gid in myrange
    I = cartinds[gid]
    if isdirichlet(grid,I)
      gridsol[I] = dirichlet(grid, I)
    end
  end

  sync(gridsol)

  @test check_solution(gridsol, grid) == 0
  println("-----------------------------------")
  println("test time:")
  @time check_solution(gridsol, grid)
  @time check_solution(gridsol, grid)
  @time check_solution(gridsol, grid)

  return (grid,gridsol)
end

(grid,sol) = solve_laplace2d(comm)

# if my_rank == 0
#   using Plots
#   heatmap(xrange(grid),yrange(grid),getblock(sol[:,:]))
#   gui()
#   readline()
# end
