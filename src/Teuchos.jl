module Teuchos
using CxxWrap, MPI
import ..libjltrilinos
import ..CxxUnion
import Base.get

export ParameterList

abstract type PLAssociative <: CxxWrap.CppAssociative{String, Any} end

struct ArrayView{T}
  array::Ptr{T}
  size::Int
end

ArrayView(a::AbstractArray{T,1}) where {T} = ArrayView(pointer(a),length(a))
ArrayView(a::AbstractArray{T,1}, length::Integer) where {T} = ArrayView(pointer(a),length)
Base.convert(::Type{ArrayView{T}}, a::AbstractArray{T,1}) where {T} = ArrayView(a, length(a))

@readmodule(libjltrilinos, :register_teuchos)
@wraptypes

CxxWrap.argument_overloads(t::Type{ArrayView{T}}) where {T} = [AbstractArray{T,1}]

@wrapfunctions

const ParUnion = Union{CxxWrap.SmartPointer{ParameterList}, ParameterList}

get(::Type{Nothing}, ::ParUnion, ::String) = "nullptr"

# High-level interface for ParameterList
Base.length(pl::ParUnion) = numParams(pl)
function Base.getindex(pl::ParUnion, key)
  if !isParameter(pl, key)
    throw(KeyError(key))
  end
  if isSublist(pl, key)
    return sublist(pl, key)
  end
  return get(get_type(pl, key), pl, key)
end
Base.setindex!(pl::ParUnion, v, key) = set(pl, key, v)
Base.setindex!(pl::ParUnion, v::Int64, key) = set(pl, key, Int32(v))
Base.setindex!(pl::ParUnion, v::CppEnum, key) = set(pl, key, convert(Int32,v))
Base.keys(pl::ParUnion) = keys(pl)

function Base.iterate(pl::ParUnion, state=(1, keys(pl)))
  idx, keys = state

  if idx >= length(keys)
    return nothing
  end

  return ((keys[idx], pl[keys[idx]]), (idx+1,keys))
end

Base.haskey(pl::ParUnion, key) = isParameter(pl, key)
Base.isempty(pl::ParUnion) = (numParams(pl) == 0)

get(pl::ParUnion, key, default_value) = isParameter(pl, key) ? pl[key] : default_value

"""
A pair of ParameterLists, one for input and one for storing the actually used parameters and any added default values
"""
struct ParameterListPair
  input
  output::ParUnion
end

function sublist(pl::ParameterListPair, name)
  param_val = get(pl.input, name, Teuchos.ParameterList(name))
  return ParameterListPair(param_val, Teuchos.sublist(pl.output, name))
end

function get(pl::ParameterListPair, key, default_value)
  result = get(pl.input, key, default_value)
  pl.output[key] = result
  return result
end

# Interface to Array
Teuchos.Array(nelems::Integer, elem::T) where {T} = Teuchos.Array{T}(nelems, elem)
Teuchos.Array(nelems::Integer, elem::T) where {T <: AbstractString} = Teuchos.Array{AbstractString}(nelems, elem)
Base.IndexStyle(::Teuchos.Array) = IndexLinear()
Base.size(arr::Teuchos.Array) = (size(arr),)
Base.getindex(arr::Teuchos.Array, i::Integer) = at(arr,i-1)

# Convert Teuchos Comm to MPI.Comm
MPI.Comm(comm::CxxUnion{Teuchos.MpiComm}) = MPI.Comm(getRawMpiComm(comm))

end
