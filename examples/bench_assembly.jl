using MPI
using Trilinos
using Test

function build_graph!(graph)
  rowmap = Tpetra.getRowMap(graph)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)
  gmin = Tpetra.getMinGlobalIndex(rowmap)
  gmax = Tpetra.getMaxGlobalIndex(rowmap)

  row_indices = zeros(Int,5)
  row_indices_view = Teuchos.ArrayView(row_indices)

  for i in 0:n_my_elms-1
    global_row =  Tpetra.getGlobalElement(rowmap,i)
    for j in 1:5
      row_indices[j] = global_row - (3-j)
      if row_indices[j] < gmin || row_indices[j] > gmax
        row_indices[j] = global_row
      end
    end
      Tpetra.insertGlobalIndices(graph, global_row, row_indices)
  end
end

function fill_matrix!(A)
  rowmap = Tpetra.getRowMap(A)
  n_my_elms = Int(Tpetra.getNodeNumElements(rowmap))

  row_indices_arr = zeros(Int32,5)
  row_values_arr = zeros(5)

  row_indices = Teuchos.ArrayView(row_indices_arr)
  row_values = Teuchos.ArrayView(row_values_arr)
  n_elems = Ref(UInt(0))

  Tpetra.resumeFill(A)
  for row in 0:n_my_elms-1
    Tpetra.getLocalRowCopy(A, row, row_indices, row_values, n_elems)
    for j in 1:n_elems[]
      row_values_arr[j] = j
    end
    Tpetra.replaceLocalValues(A, row, row_indices, row_values)
  end
end

"""
Assemble the 2D Poisson problem on a structured grid
"""
function bench_assembly(comm, nnodes)
  # Construct map
  rowmap = Tpetra.Map(nnodes, 0, comm)

  # Construct graph (i.e. the sparsity pattern)
  matrix_graph = Tpetra.CrsGraph(rowmap, 5)
  println("-----------------------------------")
  println("Graph construction time:")
  @time build_graph!(matrix_graph)
  Tpetra.fillComplete(matrix_graph)

  # Matrix construction
  A = Tpetra.CrsMatrix(matrix_graph)
  Tpetra.resumeFill(A)
  println("Matrix fill time:")
  @time fill_matrix!(A)
  Tpetra.fillComplete(A)

  # This prints the matrix if uncommented
  #Tpetra.describe(A, Teuchos.VERB_EXTREME)
end

# MPI setup
if !MPI.Initialized()
  MPI.Init()
  MPI.finalize_atexit()
end
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

my_rank = Teuchos.getRank(comm)
if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

N = 1000000
bench_assembly(comm, N)
bench_assembly(comm, N)
bench_assembly(comm, N)
