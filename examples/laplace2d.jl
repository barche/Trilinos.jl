using MPI
using Trilinos
using Base.Test
using Compat

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
  @assert n_my_elms == length(b_view)
  for i in 1:n_my_elms
    gid = Tpetra.getGlobalElement(rowmap,i-1)
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
    global_row = Tpetra.getGlobalElement(rowmap,i)
    row_n_elems = laplace2d_indices!(row_indices, global_row, g)
    Tpetra.insertGlobalIndices(crsgraph, global_row, Teuchos.ArrayView(row_indices,row_n_elems))
  end
end

function fill_laplace2d!(A, g::CartesianGrid)
  rowmap = Tpetra.getRowMap(A)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)

  # storage for the per-row values
  row_indices = [0,0,0,0,0]
  row_values = [4.0,-1.0,-1.0,-1.0,-1.0]

  for i in 0:n_my_elms-1
    global_row = Tpetra.getGlobalElement(rowmap,i)
    row_n_elems = laplace2d_indices!(row_indices, global_row, g)
    row_values[1] = 4.0 - (5-row_n_elems)
    Tpetra.replaceGlobalValues(A, global_row, Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
  end
end

# Ensure a dirichlet condition (1 at the diagonal) for the boundaries
function set_dirichlet!(A, b, g::CartesianGrid)
  rowmap = Tpetra.getRowMap(A)
  row_indices = [0,0,0,0,0]
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
    if Tpetra.isNodeGlobalElement(rowmap, gid)
      row_n_elems = laplace2d_indices!(row_indices, gid, g)
      Tpetra.replaceGlobalValues(A, gid, Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
      b_view[Tpetra.getLocalElement(rowmap, gid)+1] = (1-x^2)*(1-y^2)
    end
  end
end

function check_solution(sol, g::CartesianGrid)
  solmap = Tpetra.getMap(sol)
  solview = Tpetra.device_view(sol)
  for i in 1:length(solview)
    gid = Tpetra.getGlobalElement(solmap, i-1)
    (x,y) = coordinates(g,gid)
    @test solview[i] ≈ (1-x^2)*(1-y^2)
  end
end

"""
Assemble the 2D Posson problem on a structured grid
"""
function laplace2d(comm, g::CartesianGrid)
  # Construct map
  rowmap = Tpetra.Map(nnodes(g), 0, comm)

  # Construct graph (i.e. the sparsity pattern)
  matrix_graph = Tpetra.CrsGraph(rowmap, 0)
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
  println("Matrix fill time:")
  @time fill_laplace2d!(A,g)
  println("Dirichlet time:")
  @time set_dirichlet!(A,b,g)
  Tpetra.fillComplete(A)

  # This prints the matrix if uncommented
  # Tpetra.describe(A, Teuchos.VERB_EXTREME)

  pl = Teuchos.ParameterList()
  pl["Solver Type"] = "Block GMRES"
  solver_types = Teuchos.sublist(pl, "Solver Types")
  solver_pl = Teuchos.sublist(solver_types, "Block GMRES")
  solver_pl["Convergence Tolerance"] = 1e-12
  solver_pl["Maximum Iterations"] = Int32(1000)
  solver_pl["Num Blocks"] = Int32(1000)
  solver_pl["prec"] = Int32(1000)

  lows = Thyra.LinearOpWithSolve(A, pl, Teuchos.VERB_MEDIUM)

  return (lows,b)
end

function solve_laplace2d(comm)

  grid = CartesianGrid(101,41,1/50)

  (lows,b) = laplace2d(comm, grid)
  (lows,b) = laplace2d(comm, grid)
  (lows,b) = laplace2d(comm, grid)

  # Compute the solution
  sol = lows \ b

  @time check_solution(sol, grid)
  @time check_solution(sol, grid)
  @time check_solution(sol, grid)

  x = linspace(-1.0,1.0,grid.nx)
  y = linspace(-1.0,1.0,grid.ny)*grid.h*(grid.ny-1)/2
  return (x,y,sol)
end

# MPI setup
MPI.Init()
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

(x,y,φ) = solve_laplace2d(comm)

if isinteractive()
using Plots
# sol2d = zeros(grid.ny,grid.nx)
# for i in 1:length(sol_view)
#   gid = Tpetra.getGlobalElement(sol_map, i-1)
#   (ix,iy) = xyindices(grid,gid)
#   sol2d[iy+1,ix+1] = sol_view[i]
# end

# gr()
# display(heatmap(x,y,φ,aspect_ratio=1))

end

MPI.Finalize()
