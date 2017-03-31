module Teuchos
using CxxWrap, MPI
import .._l_trilinos_wrap

export ParameterList

abstract PLAssociative <: CxxWrap.CppAssociative{String, Any}

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

# High-level interface for ParameterList
Base.length(pl::ptrunion(ParameterList)) = numParams(pl)
function Base.getindex(pl::ptrunion(ParameterList), key)
  if !isParameter(pl, key)
    throw(KeyError(key))
  end
  if isSublist(pl, key)
    return sublist(pl, key)
  end
  return get(get_type(pl, key), pl, key)
end
Base.setindex!(pl::ptrunion(ParameterList), v, key) = set(pl, key, v)
Base.keys(pl::ptrunion(ParameterList)) = keys(pl)
Base.start(pl::ptrunion(ParameterList)) = (start(keys(pl)), keys(pl))
Base.next(pl::ptrunion(ParameterList), state) = ((state[2][state[1]], pl[state[2][state[1]]]), (state[1]+1,state[2]))
Base.done(pl::ptrunion(ParameterList), state) = state[1] == length(state[2])+1

end
