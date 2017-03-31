module Tpetra
using CxxWrap, MPI
import .._l_trilinos_wrap
import ..CxxUnion
import ..Teuchos
import ..Kokkos

registry = load_modules(_l_trilinos_wrap)

wrap_module_types(registry)

wrap_module_functions(registry)

Base.dot{ST,LT,GT,NT}(a::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}, b::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}) = Tpetra.dot(a,b)

Map(num_indices::Integer, index_base::Integer, comm::CxxWrap.SmartPointer{Teuchos.Comm}) = Map(num_indices, index_base, comm, Kokkos.default_node_type())
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
Gets an AbstractArray-compatible device view of the whole multivector. Calls getLocalView internally.
"""
device_view{ST,LT,GT,NT}(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}) = _get_view(mv, device_view_type(mv))

"""
Gets an AbstractArray-compatible host view of the whole multivector. Calls getLocalView internally.
"""
host_view{ST,LT,GT,NT}(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}) = _get_view(mv, host_view_type(mv))

"""
Gets an AbstractArray-compatible device view of the whole vector. Calls getLocalView internally.
"""
device_view{ST,LT,GT,NT}(v::CxxUnion{Vector{ST,LT,GT,NT}}) = _get_view(v, device_view_type(v))

"""
Gets an AbstractArray-compatible host view of the whole vector. Calls getLocalView internally.
"""
host_view{ST,LT,GT,NT}(v::CxxUnion{Vector{ST,LT,GT,NT}}) = _get_view(v, host_view_type(v))


#internal methods
_get_view{ST,LT,GT,NT}(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}, view_type) = Kokkos.View(ST, Val{2}, getLocalView(view_type, mv))
_get_view{ST,LT,GT,NT}(mv::CxxUnion{Vector{ST,LT,GT,NT}}, view_type) = Kokkos.View(ST, Val{1}, getLocalView(view_type, mv))

typealias PtrViewTypes{ArrayT,DeviceT} Union{Type{Kokkos.View3{ArrayT, Kokkos.LayoutLeft, DeviceT}}, Type{Kokkos.View4{ArrayT, Kokkos.LayoutLeft, DeviceT, Void}}}
function _get_view{ST,LT,GT,NT,ArrayT,DeviceT}(mv::CxxUnion{MultiVector{ST,LT,GT,NT}}, view_type::PtrViewTypes{ArrayT,DeviceT})
  localview = getLocalView(view_type, mv)
  sizes = (Int(Kokkos.dimension(localview,0)),Int(Kokkos.dimension(localview,1)))
  return Kokkos.View(ST, Val{2}, Kokkos.PtrWrapper(Kokkos.ptr_on_device(localview), sizes))
 end
function _get_view{ST,LT,GT,NT,ArrayT,DeviceT}(mv::CxxUnion{Vector{ST,LT,GT,NT}}, view_type::PtrViewTypes{ArrayT,DeviceT})
  localview = getLocalView(view_type, mv)
  sizes = (Int(Kokkos.dimension(localview,0)),)
  return Kokkos.View(ST, Val{1}, Kokkos.PtrWrapper(Kokkos.ptr_on_device(localview), sizes))
end

end
