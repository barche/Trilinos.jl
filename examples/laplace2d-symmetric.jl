using MPI
using Trilinos
using Base.Test
using Compat

if isinteractive()
  using Plots
end

const IdxT = Int

struct CartesianGrid
  nx::Int # Number of points in the X direction
  ny::Int # Number of points in the Y direction
  h::Float64 # Spacing
end

"""
List of all the nodes in the mesh
"""
globalnodes(g::CartesianGrid) = CartesianRange((g.nx, g.ny))

"""
Number of nodes in the mesh
"""
numglobalnodes(g::CartesianGrid) = g.nx*g.ny

"""
2D indices in the cartesian grid (1-based)
"""
@inline function xyindices(g::CartesianGrid, i)
  ix = i % g.nx
  iy = (i-ix) ÷ g.nx
  return CartesianIndex(ix+1,iy+1)
end

"""
Global linear index into the grid (1-based)
"""
@inline globalindex(g, I::CartesianIndex)::Int = (I[1]-1)+(I[2]-1)*g.nx + 1

"""
Coordinates of grid nodes
"""
function coordinates(g::CartesianGrid, I::CartesianIndex)
  x_center = g.h*(g.nx-1)/2
  y_center = g.h*(g.ny-1)/2
  return ((I[1]-1)*g.h - x_center, (I[2]-1)*g.h - y_center)
end

"""
Grid global indices for the 2D laplace stencil (1-based)
"""
function laplace2d_indices!(out_indices, g::CartesianGrid, I)
  n_inds = 1
  out_indices[n_inds] = globalindex(g, I)
  ix = I[1]
  iy = I[2]
  if iy != g.ny
      n_inds += 1
      out_indices[n_inds] = globalindex(g, CartesianIndex(ix,iy+1))
  end
  if ix != 1
      n_inds += 1
      out_indices[n_inds] = globalindex(g, CartesianIndex(ix-1,iy))
  end
  if iy != 1
      n_inds += 1
      out_indices[n_inds] = globalindex(g, CartesianIndex(ix,iy-1))
  end
  if ix != g.nx
      n_inds += 1
      out_indices[n_inds] = globalindex(g, CartesianIndex(ix+1,iy))
  end
  return n_inds
end

abstract type IndexMapping end

struct MappingArray <: IndexMapping
  array::Array{Int,1}
  grid::CartesianGrid
end

Base.getindex(m::IndexMapping, i) = m.array[i]
Base.getindex(m::IndexMapping, I::CartesianIndex) = m.array[globalindex(m.grid,I)]
Base.in(i, m::IndexMapping) = m.array[i] != -1

"""
Make a mapping between grid GID and linear system GIDs
"""
function MappingArray(grid, isdirifunc)
  mappingarray = fill(-1, nnodes(grid))
  gid = 0
  for I in eachindex(g)
    if isdirifunc(grid, I)
      continue
    end
    mappingarray[globalindex(grid,I)] = gid
    gid += 1
  end
  return MappingArray(mappingarray, grid)
end

"""
Map indices in-place from grid GID to linear system GID, putting mapped indices at the beginning of the supplied array and returning the number of
actually mapped indices
"""
function maptosystem!(indices, n, mapping)
  nmapped = 1
  for i in 1:n
    if indices[i] ∈ mapping
      indices[nmapped] = mapping[indices[i]]
      nmapped += 1
    end
  end
  return nmapped
end

"""
Filter out dirichlet conditions and map indices to the system, keeping only values that belong in the system. Returns a pair with: 
  - the number of retained nodes for the system
  - the sum of value[i]*dirichlet_value for any dirichlet nodes encountered
"""
function filterdirichlet!(indices, values, n, mapping, isdirichlet::Function, dirichletvalue::Function)
  nmapped = 1
  diri_val = 0.0
  for i in 1:n
    if indices[i] ∈ mapping
      if isdirichlet(indices[i])
        diri_val += dirichletvalue(indices[i])
      end
      indices[nmapped] = mapping[indices[i]]
      values[nmapped] = values[i]
      nmapped += 1
    end
  end
  return (nmapped, diri_val)
end

"""
Set the source term
"""
function set_source_term!(b, systemmapping)
  rowmap = Tpetra.getMap(b)
  b_view = Tpetra.device_view(b)
  g = systemmapping.grid

  for I in eachindex(g)
    (x,y) = coordinates(g,I)
    b_view[Tpetra.getLocalElement(rowmap,systemmapping[I])] = 2*g.h^2*((1-x^2)+(1-y^2))
  end

end

function graph_laplace2d!(crsgraph, systemmapping)
  rowmap = Tpetra.getRowMap(crsgraph)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)
  g = systemmapping.grid

  # storage for the per-row indices
  row_indices = [0,0,0,0,0]

  for I in eachindex(g)
    grid_gid = globalindex(g, I)
    if grid_gid ∈ systemmapping
      grid_n_elems = laplace2d_indices!(row_indices, grid_gid, g)
      row_n_elems = maptosystem!(row_indices, grid_n_elems, systemmapping)
      Tpetra.insertGlobalIndices(crsgraph, systemmapping[grid_gid], Teuchos.ArrayView(row_indices,row_n_elems))
    end
  end
end

function fill_laplace2d!(A, g::CartesianGrid)
  rowmap = Tpetra.getRowMap(A)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)

  # storage for the per-row values
  row_indices = IdxT[0,0,0,0,0]
  row_values = [4.0,-1.0,-1.0,-1.0,-1.0]

  for I in eachindex(g)
    grid_gid = globalindex(g, I)
    if grid_gid ∈ systemmapping
      grid_n_elems = laplace2d_indices!(row_indices, grid_gid, g)
      row_n_elems = maptosystem!(row_indices, grid_n_elems, systemmapping)
      Tpetra.replaceGlobalValues(A, systemmapping[grid_gid], Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
    end
  end
end

# Ensure a dirichlet condition (1 at the diagonal) for the boundaries
function set_dirichlet!(A, b, systemmapping)
  rowmap = Tpetra.getRowMap(A)
  row_indices = IdxT[0,0,0,0,0]
  row_values = [1.0,0.0,0.0,0.0,0.0]
  g = systemmapping.grid

  b_view = Tpetra.device_view(b)

  # apply
  for grid_gid in dirichletnodes(g, isdirichlet)
    (x,y) = coordinates(g,xyindices(grid_gid))
    sys_gid = systemmapping[grid_gid]
    if Tpetra.isNodeGlobalElement(rowmap, sys_gid)
      row_n_elems = laplace2d_indices!(row_indices, gid, g)
      Tpetra.replaceGlobalValues(A, gid, Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
      b_view[Tpetra.getLocalElement(rowmap, gid)] = (1-x^2)*(1-y^2)
    end
  end
end

@inline function isdirichlet(grid::CartesianGrid, I::CartesianIndex)
  ix = I[1]
  iy = I[2]
  return ix == 0 || iy == 0 || ix == (grid.nx-1) || iy == (grid.ny-1)
end

isdirichlet(grid, gid) = isdirichlet(grid, xyindices(grid,gid))

function dirichletnodes(grid, isdirichlet::Function)
  nodes = Int[]
  for I in eachindex(grid)
    if isdirichlet(grid, I)
      push!(nodes,globalindex(grid, I))
    end
  end
end

function check_solution(sol, g::CartesianGrid)
  solmap = Tpetra.getMap(sol)
  solview = Tpetra.device_view(sol)
  result = 0
  for i in linearindices(solview)
    gid = Tpetra.getGlobalElement(solmap, i)
    (x,y) = coordinates(g,gid)
    result +=  abs(solview[i] - (1-x^2)*(1-y^2)) > 1e-10
  end
  return result == 0
end

"""
Assemble the 2D Posson problem on a structured grid
"""
function laplace2d(comm, g::CartesianGrid)
  # Construct map

  dirinodes = dirichletnodes(g, isdirichlet)

  rowmap = Tpetra.Map(nnodes(g), 0, comm)

  # Construct graph (i.e. the sparsity pattern)
  matrix_graph = Tpetra.CrsGraph(rowmap, 0)
  println("-----------------------------------")
  println("Graph construction time:")
  @time graph_laplace2d!(matrix_graph, g)
  Tpetra.fillComplete(matrix_graph)

  # Construct the vectors
  b = Tpetra.Vector(Tpetra.getRangeMap(matrix_graph))
  println("Source term time:")
  @time set_source_term!(b,g)

  # Matrix construction
  A = Tpetra.CrsMatrix(matrix_graph)
  Tpetra.resumeFill(A)
  @test Tpetra.isLocallyIndexed(A)
  println("Matrix fill time:")
  @time fill_laplace2d!(A,g)
  println("Dirichlet time:")
  @time set_dirichlet!(A,b,g)
  Tpetra.fillComplete(A)
  @test Tpetra.isLocallyIndexed(A)

  # This prints the matrix if uncommented
  # Tpetra.describe(A, Teuchos.VERB_EXTREME)

  params = Trilinos.default_parameters()
  maxiter = 1000
  solver_params = params["Linear Solver Types"]["Belos"]["Solver Types"]["BLOCK GMRES"]
  solver_params["Convergence Tolerance"] = 1e-12
  solver_params["Verbosity"] = Belos.StatusTestDetails + Belos.FinalSummary + Belos.TimingDetails
  solver_params["Num Blocks"] = Int32(maxiter)
  solver_params["Maximum Iterations"] = Int32(maxiter)

  #params["Preconditioner Type"] = "None"
  
  solver = TpetraSolver(A, params)

  return (solver,b)
end

function solve_laplace2d(comm)

  nx = 11
  grid = CartesianGrid(nx,nx,2/(nx-1))

  sysmap = makesystemmap(grid, isdirichlet)
  println(sysmap)
  @show length(sysmap)

  exit()

  (lows,b) = laplace2d(comm, grid)
  (lows,b) = laplace2d(comm, grid)
  (lows,b) = laplace2d(comm, grid)

  # Compute the solution
  sol = lows \ b

  @test check_solution(sol, grid)
  println("-----------------------------------")
  println("test time:")
  @time check_solution(sol, grid)
  @time check_solution(sol, grid)
  @time check_solution(sol, grid)

  x = linspace(-1.0,1.0,grid.nx)
  y = linspace(-1.0,1.0,grid.ny)*grid.h*(grid.ny-1)/2
  return (x,y,Tpetra.device_view(sol),grid)
end

# MPI setup
if !MPI.Initialized()
  MPI.Init()
end
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

my_rank = Teuchos.getRank(comm)
if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

nb_procs = Teuchos.getSize(comm)
if USE_LOCAL_INDICES && nb_procs > 1
  error("Local indices may only be used on single core execution")
end

(x,y,f) = solve_laplace2d(comm)

if isinteractive()
  plot(x, y, f, aspect_ratio=1, seriestype=:heatmap)
else
  MPI.Finalize()
end