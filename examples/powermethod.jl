# Powermethod using Tpetra
# Adapted from https://trilinos.org/docs/dev/packages/tpetra/doc/html/Tpetra_Lesson03.html

using MPI
using Trilinos

function powermethod(A, niters::Number, tolerance::Number)
  q = Tpetra.Vector(Tpetra.getDomainMap(A))
  z = Tpetra.Vector(Tpetra.getRangeMap(A))
  resid = Tpetra.Vector(Tpetra.getRangeMap(A))

  Tpetra.randomize(z)

  lambda = 0.0
  normz = 0.0
  residual = 0.0

  report_frequency = 10
  for iter in 1:niters
    normz = Tpetra.norm2(z)         # Compute the 2-norm of z
    Tpetra.scale(q, 1.0 / normz, z) # q := z / normz
    Tpetra.apply(A, q, z)           # z := A * q
    lambda = dot(q,z)               # Approx. max eigenvalue
    # Compute and report the residual norm every report_frequency
    # iterations, or if we've reached the maximum iteration count.
    if iter % report_frequency == 0 || iter == 1
      Tpetra.update(resid, 1.0, z, -lambda, q, 0.0) # z := A*q - lambda*q
      residual = Tpetra.norm2(resid)                     # 2-norm of the residual vector

      println("Iteration $iter:")
      println("- lambda = $lambda")
      println("- ||A*q - lambda*q||_2 = $residual")
    end
    if residual < tolerance
      println("Converged after $iter iterations")
      break
    elseif (iter == niters)
      println("Failed to converge after $niters iterations")
      break
    end
  end

  return lambda
end

# MPI setup
MPI.Init()
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

# Communication map
const num_global_indices = 50
const indexbase = 0
map = Tpetra.Map(num_global_indices, indexbase, comm)

# Number of local elements
my_rank = Teuchos.getRank(comm)
# if my_rank != 0
#   redirect_stdout(DevNull)
#   redirect_stderr(DevNull)
# end

num_my_elements = Int(Tpetra.getNodeNumElements(map)) # UInt otherwise
println("Number of elements for rank $my_rank is $num_my_elements")

# Construct the matrix
A = Tpetra.CrsMatrix(map, 0)

for local_row in 0:num_my_elements-1
  global_row = Tpetra.getGlobalElement(map, local_row)
  if global_row == 0
    Tpetra.insertGlobalValues(A, global_row, [global_row, global_row+1], [2.0, -1.0])
  elseif global_row == num_my_elements-1
    Tpetra.insertGlobalValues(A, global_row, [global_row-1, global_row], [-1.0, 2.0])
  else
    Tpetra.insertGlobalValues(A, global_row, [global_row-1, global_row, global_row+1], [-1.0, 2.0, -1.0])
  end
end

Tpetra.fillComplete(A)

# Number of iterations
niters = 500
# Target tolerance
tolerance = 1e-2

lambda = powermethod(A, niters, tolerance)
println("\nMaximal estimated eigenvalue: $lambda\n")

println("Increasing magnitude of A(0,0), solving again")

Tpetra.resumeFill(A)

if Tpetra.isNodeGlobalElement(Tpetra.getRowMap(A), 0)
  first_row_id = 0
  num_entries_in_row = Tpetra.getNumEntriesInGlobalRow(A, first_row_id)

  row_values = Vector{Float64}(num_entries_in_row)
  row_indices = Vector{Int}(num_entries_in_row)

  num_entries_returned = Ref(UInt(0))

  # The last 3 parameters here are output parameters. The ArrayView wraps the provided Julia array, which must be at least as large as the number of entries stored
  # num_entries_returned contains the actual number of items that is read
  Tpetra.getGlobalRowCopy(A, first_row_id, Teuchos.ArrayView(row_indices), Teuchos.ArrayView(row_values), num_entries_returned)

  for i in 1:num_entries_returned[]
    if row_indices[i] == first_row_id
      row_values[i] *= 10.0
    end
  end

  # Conversion to ArrayView is implicit here, because the arrays are read-only
  Tpetra.replaceGlobalValues(A, first_row_id, row_indices, row_values)
end

Tpetra.fillComplete(A)

lambda = powermethod(A, niters, tolerance)
println("\nMaximal estimated eigenvalue: $lambda\n")

MPI.Finalize()
