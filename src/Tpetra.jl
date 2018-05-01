module Tpetra
using CxxWrap, MPI
using Compat
import .._l_trilinos_wrap
import ..CxxUnion
import ..Teuchos
import ..Kokkos

wrap_module(_l_trilinos_wrap, Tpetra)

Base.dot{ST,LT,GT,NT}(a::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}, b::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}) = Tpetra.dot(a,b)

Map(num_indices::Integer, index_base::Integer, comm::CxxWrap.SmartPointer{<:Teuchos.Comm}) = Map(num_indices, index_base, comm, Kokkos.default_node_type())
Map(num_indices::Integer, index_list, index_base::Integer, comm::CxxWrap.SmartPointer{<:Teuchos.Comm}) = Map(num_indices, index_list, index_base, comm, Kokkos.default_node_type())
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

"""
CrsMatrix apply function with defauls arguments
"""
apply{ST,LT,GT,NT}(A::CxxUnion{CrsMatrix{ST,LT,GT,NT}},
                   X::CxxUnion{MultiVector{ST,LT,GT,NT}},
                   Y::CxxUnion{MultiVector{ST,LT,GT,NT}},
                   mode::Teuchos.ETransp=Teuchos.NO_TRANS,
                   α::Number=1,
                   β::Number=0) = _apply(A,X,Y,mode,α,β)

"""
Gets an AbstractArray-compatible device view of the local part of the multivector. Calls getLocalView internally.
"""
device_view{ST,LT,GT,NT}(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}) = Kokkos.view(getLocalView(device_view_type(mv),mv),Val{2})

"""
Gets an AbstractArray-compatible host view of the local part of the multivector. Calls getLocalView internally.
"""
host_view{ST,LT,GT,NT}(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}) = Kokkos.view(getLocalView(host_view_type(mv),mv),Val{2})

"""
Gets an AbstractArray-compatible device view of the local part of the vector. Calls getLocalView internally.
"""
device_view{ST,LT,GT,NT}(v::CxxUnion{Vector{ST,LT,GT,NT}}) = Kokkos.view(getLocalView(device_view_type(v),v),Val{1})

"""
Gets an AbstractArray-compatible host view of the local part of the vector. Calls getLocalView internally.
"""
host_view{ST,LT,GT,NT}(v::CxxUnion{Vector{ST,LT,GT,NT}}) = Kokkos.view(getLocalView(host_view_type(v),v),Val{1})

function CrsGraph(rowmap::CxxUnion{Map{LT,GT,NT}}, colmap::CxxUnion{Map{LT,GT,NT}}, graph::Kokkos.StaticCrsGraph) where {LT,GT,NT}
  return CrsGraph(rowmap, colmap, local_graph_type(CrsGraph{LT,GT,NT})(graph))
end

end
