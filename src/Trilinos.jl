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

# Base type for all RCP-wrappable types in Julia
abstract JuliaRCP

# Generate RCP overloads automatically
CxxWrap.argument_overloads{T <: JuliaRCP}(t::Type{T}) = [Trilinos.Teuchos.RCP{T}]

registry = load_modules(_l_trilinos_wrap)

module Teuchos
using Trilinos, CxxWrap, MPI

wrap_module(Trilinos.registry)

end

module Tpetra
using Trilinos, CxxWrap, MPI

wrap_module(Trilinos.registry)

end

end # module
