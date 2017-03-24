using MPI
using Trilinos
using Base.Test

"""
Matrix global indices for the 2D laplace stencil (0-based!)
"""
function laplace2d_indices!(inds_array,i,nx,ny)
  ix = i % nx
  iy = (i-ix) ÷ nx

  n_inds = 1
  inds_array[n_inds] = i
  if iy != ny-1
      n_inds += 1
      inds_array[n_inds] = i+nx
  end
  if ix != 0
      n_inds += 1
      inds_array[n_inds] = i-1
  end
  if iy != 0
      n_inds += 1
      inds_array[n_inds] = i-nx
  end
  if ix != nx-1
      n_inds += 1
      inds_array[n_inds] = i+1
  end
  return n_inds
end

function fill_laplace2d!(A_rcp, nx::Integer, ny::Integer)
  A = A_rcp[]
  rowmap = (Tpetra.getRowMap(A))[]
  n_my_elms = Tpetra.getNodeNumElements(rowmap)

  # storage for the per-row values
  row_indices = [0,0,0,0,0]
  row_values = [4.0,-1.0,-1.0,-1.0,-1.0]

  for i in 0:n_my_elms-1
    global_row = Tpetra.getGlobalElement(rowmap,i)
    row_n_elems = laplace2d_indices!(row_indices, global_row, nx, ny)
    row_values[1] = 4.0 - (5-row_n_elems)
    Tpetra.insertGlobalValues(A, global_row, Teuchos.ArrayView(row_indices,row_n_elems), Teuchos.ArrayView(row_values,row_n_elems))
  end
end

"""
Solve the 2D Posson problem on a structured grid with nx*ny nodes
"""
function laplace2d(comm, nx::Integer, ny::Integer)
  # Construct map
  nnodes = nx*ny
  rowmap = Tpetra.Map(nnodes, 0, comm)

  # Matrix construction
  A = Tpetra.CrsMatrix(rowmap, 0)
  @time fill_laplace2d!(A,nx,ny)
  Tpetra.fillComplete(A)

  #Tpetra.describe(A, Teuchos.VERB_EXTREME)

  # Construct the vectors
  x_ref = Tpetra.Vector(Tpetra.getDomainMap(A)) # reference solution
  b = Tpetra.Vector(Tpetra.getRangeMap(A))

  # Construct RHS from reference solution
  Tpetra.randomize(x_ref)
  Tpetra.apply(A,x_ref,b)

  lows = Thyra.LinearOpWithSolve(A, Teuchos.ParameterList(), Teuchos.VERB_NONE)

  # Compute the solution
  x = lows \ b

  # Check solution
  # for (xi, xi_ref) in zip(Tpetra.device_view(x), Tpetra.device_view(x_ref))
  #   @test abs(xi-xi_ref) < abs(xi)*1e-4
  # end

  return x
end

# MPI setup
MPI.Init()
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

x = laplace2d(comm, 200, 20)
x = laplace2d(comm, 200, 20)
x = laplace2d(comm, 200, 20)
MPI.Finalize()
