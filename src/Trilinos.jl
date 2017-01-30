module Trilinos

export Teuchos
export Tpetra

using CxxWrap

# Load the deps
const depsfile = joinpath(dirname(dirname(@__FILE__)), "deps", "deps.jl")
if !isfile(depsfile)
  error("$depsfile not found, package Trilinos did not build properly")
end
include(depsfile)

# Base type for RCP-wrappable types in Julia
abstract RCPWrappable

# Base type for RCP-wrappable associative types in Julia
abstract RCPAssociative <: Associative{String, Any}

# Generate RCP overloads automatically
CxxWrap.argument_overloads{T <: RCPWrappable}(t::Type{T}) = [Trilinos.Teuchos.RCP{T}]
CxxWrap.argument_overloads{T <: RCPAssociative}(t::Type{T}) = [Trilinos.Teuchos.RCP{T}]
# Overload for size_t
@static if Sys.WORD_SIZE == 64
  CxxWrap.argument_overloads(t::Type{UInt64}) = [Int64]
end

registry = load_modules(_l_trilinos_wrap)

module Teuchos
using Trilinos, CxxWrap, MPI

wrap_module(Trilinos.registry)

Base.getindex{T}(p::RCP{T}) = convert(T, p)

# High-level interface for ParameterList
Base.length(pl::ParameterList) = numParams(pl)
function Base.getindex(pl::ParameterList, key)
  if !isParameter(pl, key)
    throw(KeyError(key))
  end
  if isSublist(pl, key)
    return sublist(pl, key)
  end
  return get(get_type(pl, key), pl, key)
end
Base.setindex!(pl::ParameterList, v, key) = set(pl, key, v)
Base.keys(pl::ParameterList) = keys(pl)
Base.start(pl::ParameterList) = (start(keys(pl)), keys(pl))
Base.next(pl::ParameterList, state) = ((state[2][state[1]], pl[state[2][state[1]]]), (state[1]+1,state[2]))
Base.done(pl::ParameterList, state) = state[1] == length(state[2])+1

# Convenience methods for direct RCP access
Base.length(pl::RCP{ParameterList}) = Base.length(convert(ParameterList, pl))
Base.getindex(pl::RCP{ParameterList}, key) = Base.getindex(convert(ParameterList, pl), key)
Base.setindex!(pl::RCP{ParameterList}, v, key) = Base.setindex!(convert(ParameterList, pl), v, key)

end

module Tpetra
using Trilinos, CxxWrap, MPI

wrap_module(Trilinos.registry)

Base.dot(a::Tpetra.Vector, b::Tpetra.Vector) = Tpetra.dot(a,b)
Tpetra.getGlobalElement(map, idx::UInt64) = Tpetra.getGlobalElement(map, convert(Int, idx))

"""
Construct a Trilinos sparse matrix from a SparseMatrixCSC. Note that the resulting matrix is the transpose of the Julia matrix, since Trilinos is row-based.
The resulting matrix has a fixed structure (using a constant CrsGraph in Tpetra) and its rows are distributed over the available MPI processes.
The matrix A is supposed to be completely supplied on each rank.
"""
function CrsMatrix{ScalarT, GlobalOrdninalT}(A::SparseMatrixCSC{ScalarT, GlobalOrdninalT}, rowmap)
  # Create the sparse matrix structure
  matrix_graph = CrsGraph(rowmap, GlobalOrdninalT(0))
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

end # module
