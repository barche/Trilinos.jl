module Tpetra
using CxxWrap, MPI
import .._l_trilinos_wrap
import ..RCPWrappable
import ..RCPAssociative
import ..Teuchos
import ..Kokkos

registry = load_modules(_l_trilinos_wrap)

wrap_module_types(registry)

CxxWrap.argument_overloads{T1,T2,T3,T4}(t::Type{Teuchos.RCP{Tpetra.Operator{T1,T2,T3,T4}}}) = [Teuchos.RCP{Tpetra.CrsMatrix{T1,T2,T3,T4}}]
CxxWrap.argument_overloads{T1,T2,T3,T4}(t::Type{Tpetra.MultiVector{T1,T2,T3,T4}}) = [Teuchos.RCP{Tpetra.Vector{T1,T2,T3,T4}},Teuchos.RCP{Tpetra.MultiVector{T1,T2,T3,T4}}]

wrap_module_functions(registry)

Base.dot(a::Tpetra.Vector, b::Tpetra.Vector) = Tpetra.dot(a,b)
Base.dot(a::Teuchos.RCP, b::Teuchos.RCP) = Tpetra.dot(a,b)

Map(num_indices::Integer, index_base::Integer, comm::Teuchos.RCP{Teuchos.Comm}) = Map(num_indices, index_base, comm, Kokkos.default_node_type())
getGlobalElement(map, idx::UInt64) = getGlobalElement(map, convert(Int, idx))

"""
Construct a Trilinos sparse matrix from a SparseMatrixCSC. Note that the resulting matrix is the transpose of the Julia matrix, since Trilinos is row-based.
The resulting matrix has a fixed structure (using a constant CrsGraph in Tpetra) and its rows are distributed over the available MPI processes.
The matrix A is supposed to be completely supplied on each rank.
"""
function CrsMatrix{ScalarT, GlobalOrdinalT}(A::SparseMatrixCSC{ScalarT, GlobalOrdinalT}, rowmap)
  # Create the sparse matrix structure
  matrix_graph = CrsGraph(rowmap, GlobalOrdinalT(0))
  rows = rowvals(A)
  vals = nonzeros(A)
  num_my_elements = getNodeNumElements(rowmap)
  for local_row = 0:num_my_elements-1
    global_row = getGlobalElement(rowmap, local_row)
    insertGlobalIndices(matrix_graph, global_row, rows[nzrange(A, global_row+1)]-1)
  end
  fillComplete(matrix_graph)

  tpetra_matrix = CrsMatrix(matrix_graph)
  resumeFill(tpetra_matrix)
  for local_row = 0:num_my_elements-1
    global_row = getGlobalElement(rowmap, local_row)
    replaceGlobalValues(tpetra_matrix, global_row, rows[nzrange(A, global_row+1)]-1, vals[nzrange(A, global_row+1)])
  end
  fillComplete(tpetra_matrix)
  return tpetra_matrix
end

end
