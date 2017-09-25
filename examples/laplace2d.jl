using MPI
using Trilinos
using Base.Test
using Compat

const USE_LOCAL_INDICES = !isempty(ARGS) && ARGS[1] == "local"

@static if USE_LOCAL_INDICES
  @compat const IdxT = Int32
else
  @compat const IdxT = Int
end

function replace_values(A, gid, indices, values)
  @static if USE_LOCAL_INDICES
    return Tpetra.replaceLocalValues(A, gid, indices, values)
  else
    return Tpetra.replaceGlobalValues(A, gid, indices, values)
  end
end

function global_element(rowmap,i)
  @static if USE_LOCAL_INDICES
    return Int(i)
  else
    return Tpetra.getGlobalElement(rowmap,i)
  end
end

function local_element(rowmap,gid)
  @static if USE_LOCAL_INDICES
    return gid
  else
    return Tpetra.getLocalElement(rowmap,gid)
  end
end

function is_node_global_element(rowmap,gid)
  @static if USE_LOCAL_INDICES
    return true
  else
    return Tpetra.isNodeGlobalElement(rowmap,gid)
  end
end

immutable CartesianGrid
  nx::Int # Number of points in the X direction
  ny::Int # Number of points in the Y direction
  h::Float64 # Spacing
end

nnodes(g::CartesianGrid) = g.nx*g.ny

function xyindices(g::CartesianGrid, i)
  ix = i % g.nx
  iy = (i-ix) ÷ g.nx
  return (ix,iy)
end

function coordinates(g::CartesianGrid, i)
  (ix,iy) = xyindices(g,i)
  x_center = g.h*(g.nx-1)/2
  y_center = g.h*(g.ny-1)/2
  return (ix*g.h - x_center, iy*g.h - y_center)
end

"""
Matrix global indices for the 2D laplace stencil (0-based!)
"""
function laplace2d_indices!(inds_array,i,g::CartesianGrid)
  (ix,iy) = xyindices(g,i)

  n_inds = 1
  inds_array[n_inds] = i
  if iy != g.ny-1
      n_inds += 1
      inds_array[n_inds] = i+g.nx
  end
  if ix != 0
      n_inds += 1
      inds_array[n_inds] = i-1
  end
  if iy != 0
      n_inds += 1
      inds_array[n_inds] = i-g.nx
  end
  if ix != g.nx-1
      n_inds += 1
      inds_array[n_inds] = i+1
  end
  return n_inds
end

function set_source_term!(b, g::CartesianGrid)
  rowmap = Tpetra.getMap(b)
  b_view = Tpetra.device_view(b)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)
  @assert n_my_elms == length(linearindices(b_view))
  for i in linearindices(b_view)
    gid = global_element(rowmap,i)
    (x,y) = coordinates(g,gid)
    b_view[i] = 2*g.h^2*((1-x^2)+(1-y^2))
  end
end

function graph_laplace2d!(crsgraph, g::CartesianGrid)
  rowmap = Tpetra.getRowMap(crsgraph)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)

  # storage for the per-row indices
  row_indices = [0,0,0,0,0]

  for i in 0:n_my_elms-1
    global_row = global_element(rowmap,i)
    row_n_elems = laplace2d_indices!(row_indices, global_row, g)
    Tpetra.insertGlobalIndices(crsgraph, global_row, Teuchos.ArrayView(row_indices,row_n_elems))
  end
end

function fill_laplace2d!(A, g::CartesianGrid)
  rowmap = Tpetra.getRowMap(A)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)

  # storage for the per-row values
  row_indices = IdxT[0,0,0,0,0]
  row_values = [4.0,-1.0,-1.0,-1.0,-1.0]

  for i in 0:n_my_elms-1
    global_row = global_element(rowmap,i)
    row_n_elems = laplace2d_indices!(row_indices, global_row, g)
    row_values[1] = 4.0 - (5-row_n_elems)
    replace_values(A, global_row, Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
  end
end

# Ensure a dirichlet condition (1 at the diagonal) for the boundaries
function set_dirichlet!(A, b, g::CartesianGrid)
  rowmap = Tpetra.getRowMap(A)
  row_indices = IdxT[0,0,0,0,0]
  row_values = [1.0,0.0,0.0,0.0,0.0]

  b_view = Tpetra.device_view(b)

  # Collect the boundary nodes
  boundary_gids = zeros(Int,2*(g.nx+g.ny))
  gid_idx = 1
  # left and right
  for iy in 0:(g.ny-1)
    boundary_gids[gid_idx] = iy*g.nx
    boundary_gids[gid_idx+1] = boundary_gids[gid_idx] + g.nx-1
    gid_idx += 2
  end

  # top and bottom
  for ix in 0:(g.nx-1)
    boundary_gids[gid_idx] = ix
    boundary_gids[gid_idx+1] = ix + (g.ny-1)*g.nx
    gid_idx += 2
  end

  # apply
  for gid in boundary_gids
    (x,y) = coordinates(g,gid)
    if is_node_global_element(rowmap, gid)
      row_n_elems = laplace2d_indices!(row_indices, gid, g)
      replace_values(A, gid, Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
      b_view[local_element(rowmap, gid)] = (1-x^2)*(1-y^2)
    end
  end
end

function check_solution(sol, g::CartesianGrid)
  solmap = Tpetra.getMap(sol)
  solview = Tpetra.device_view(sol)
  result = 0
  for i in linearindices(solview)
    gid = global_element(solmap, i)
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

  grid = CartesianGrid(101,101,1/50)

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
  using Plots
  heatmap(x,y,f,aspect_ratio=1)
else
  MPI.Finalize()
end