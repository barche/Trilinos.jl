module Tpetra
using CxxWrap, MPI
import ..libjltrilinos
import ..CxxUnion
import ..Teuchos
import ..Kokkos

import LinearAlgebra
using SparseArrays

@wrapmodule(libjltrilinos, :register_tpetra)

LinearAlgebra.dot(a::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}, b::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}) where {ST,LT,GT,NT} = Tpetra.dot(a,b)

Map(num_indices::Integer, index_base::Integer, comm::CxxWrap.SmartPointer{<:Teuchos.Comm}) = Map(num_indices, index_base, comm, Kokkos.default_node_type())
Map(num_indices::Integer, index_list, index_base::Integer, comm::CxxWrap.SmartPointer{<:Teuchos.Comm}) = Map(num_indices, index_list, index_base, comm, Kokkos.default_node_type())
getGlobalElement(map, idx::UInt64) = getGlobalElement(map, convert(Int, idx))

"""
Construct a Trilinos sparse matrix from a SparseMatrixCSC. Note that the resulting matrix is the transpose of the Julia matrix, since Trilinos is row-based.
The resulting matrix has a fixed structure (using a constant CrsGraph in Tpetra) and its rows are distributed over the available MPI processes.
The matrix A is supposed to be completely supplied on each rank.
"""
function CrsMatrix(A::SparseMatrixCSC{ScalarT, GlobalOrdinalT}, rowmap) where {ScalarT,GlobalOrdinalT}
  # Create the sparse matrix structure
  matrix_graph = CrsGraph(rowmap, GlobalOrdinalT(0))
  rows = rowvals(A)
  vals = nonzeros(A)
  num_my_elements = getNodeNumElements(rowmap)
  for local_row = 0:num_my_elements-1
    global_row = getGlobalElement(rowmap, local_row)
    insertGlobalIndices(matrix_graph, global_row, rows[nzrange(A, global_row+1)].-1)
  end
  fillComplete(matrix_graph)

  tpetra_matrix = CrsMatrix(matrix_graph)
  resumeFill(tpetra_matrix)
  for local_row = 0:num_my_elements-1
    global_row = getGlobalElement(rowmap, local_row)
    replaceGlobalValues(tpetra_matrix, global_row, rows[nzrange(A, global_row+1)].-1, vals[nzrange(A, global_row+1)])
  end
  fillComplete(tpetra_matrix)
  return tpetra_matrix
end

"""
CrsMatrix apply function with defauls arguments
"""
apply(A::CxxUnion{CrsMatrix{ST,LT,GT,NT}},
                   X::CxxUnion{MultiVector{ST,LT,GT,NT}},
                   Y::CxxUnion{MultiVector{ST,LT,GT,NT}},
                   mode::Teuchos.ETransp=Teuchos.NO_TRANS,
                   α::Number=1,
                   β::Number=0) where {ST,LT,GT,NT} = _apply(A,X,Y,mode,α,β)

"""
Gets an AbstractArray-compatible device view of the local part of the multivector. Calls getLocalView internally.
"""
device_view(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}) where {ST,LT,GT,NT} = Kokkos.view(getLocalView(device_view_type(mv),mv),Val{2})

"""
Gets an AbstractArray-compatible host view of the local part of the multivector. Calls getLocalView internally.
"""
host_view(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}) where {ST,LT,GT,NT} = Kokkos.view(getLocalView(host_view_type(mv),mv),Val{2})

"""
Gets an AbstractArray-compatible device view of the local part of the vector. Calls getLocalView internally.
"""
device_view(v::CxxUnion{Vector{ST,LT,GT,NT}}) where {ST,LT,GT,NT} = Kokkos.view(getLocalView(device_view_type(v),v),Val{1})

"""
Gets an AbstractArray-compatible host view of the local part of the vector. Calls getLocalView internally.
"""
host_view(v::CxxUnion{Vector{ST,LT,GT,NT}}) where {ST,LT,GT,NT} = Kokkos.view(getLocalView(host_view_type(v),v),Val{1})

function CrsGraph(rowmap::CxxUnion{Map{LT,GT,NT}}, colmap::CxxUnion{Map{LT,GT,NT}}, graph::Kokkos.StaticCrsGraph) where {LT,GT,NT}
  return CrsGraph(rowmap, colmap, Kokkos.to_cpp(graph,local_graph_type(CrsGraph{LT,GT,NT})))
end

# Serial dump to regular dense array
function to_array(A)
  nb_rows = Int(getNodeNumRows(A))
  result = zeros(nb_rows,nb_rows)
  row_indices = zeros(Int32,100)
  row_values = zeros(Float64,100)

  indices_view = Teuchos.ArrayView(row_indices)
  values_view = Teuchos.ArrayView(row_values)
  rowlength_ref = Ref(UInt(0))

  for row in 1:nb_rows
    getLocalRowCopy(A, row-1, indices_view, values_view, rowlength_ref)
    for i in 1:rowlength_ref[]
      result[row,row_indices[i]+1] = row_values[i]
    end
  end
  return result
end

end
