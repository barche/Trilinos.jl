module Teuchos
using CxxWrap, MPI
import .._l_trilinos_wrap
import ..RCPWrappable
import ..RCPAssociative

export ParameterList

immutable ArrayView{T}
  array::Array{T,1}
  size::Int
end

ArrayView{T}(a::Array{T,1}) = ArrayView(a,length(a))
Base.convert{T}(::Type{ArrayView{T}}, a::Array{T,1}) = ArrayView(a, length(a))

registry = load_modules(_l_trilinos_wrap)
wrap_module_types(registry)

CxxWrap.argument_overloads{T}(t::Type{RCPPtr{T}}) = [RCP{T}]
CxxWrap.argument_overloads{T}(t::Type{ArrayView{T}}) = [Array{T,1}]

wrap_module_functions(registry)

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
