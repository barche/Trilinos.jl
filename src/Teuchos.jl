module Teuchos
using CxxWrap, MPI
using Compat
import .._l_trilinos_wrap

export ParameterList

import Base.get

@compat abstract type PLAssociative <: CxxWrap.CppAssociative{String, Any} end

immutable ArrayView{T}
  array::Ptr{T}
  size::Int
end

ArrayView{T}(a::AbstractArray{T,1}) = ArrayView(pointer(a),length(a))
ArrayView{T}(a::AbstractArray{T,1}, length::Integer) = ArrayView(pointer(a),length)
Base.convert{T}(::Type{ArrayView{T}}, a::AbstractArray{T,1}) = ArrayView(a, length(a))

registry = load_modules(_l_trilinos_wrap)
wrap_module_types(registry)

CxxWrap.argument_overloads{T}(t::Type{ArrayView{T}}) = [AbstractArray{T,1}]

wrap_module_functions(registry)

const ParUnion = Union{CxxWrap.SmartPointer{ParameterList}, ParameterList}

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
Base.keys(pl::ParUnion) = keys(pl)
Base.start(pl::ParUnion) = (start(keys(pl)), keys(pl))
Base.next(pl::ParUnion, state) = ((state[2][state[1]], pl[state[2][state[1]]]), (state[1]+1,state[2]))
Base.done(pl::ParUnion, state) = state[1] == length(state[2])+1

get(pl::ParUnion, key, default_value) = isParameter(pl, key) ? pl[key] : default_value

"""
A pair of ParameterLists, one for input and one for storing the actually used parameters and any added default values
"""
immutable ParameterListPair
  input::CxxWrap.SmartPointer{ParameterList}
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

end
