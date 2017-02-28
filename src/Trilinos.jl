module Trilinos

export Kokkos
export Teuchos
export Tpetra
export Thyra

using CxxWrap

# Load the deps
const depsfile = joinpath(dirname(dirname(@__FILE__)), "deps", "deps.jl")
if !isfile(depsfile)
  error("$depsfile not found, package Trilinos did not build properly")
end
include(depsfile)

# Base type for RCP-wrappable types in Julia
abstract RCPWrappable <: CxxWrap.CppAny

# Base type for RCP-wrappable associative types in Julia
abstract RCPAssociative <: CxxWrap.CppAssociative{String, Any}

# Generate RCP overloads automatically
CxxWrap.argument_overloads{T <: RCPWrappable}(t::Type{T}) = [Trilinos.Teuchos.RCP{T}]
CxxWrap.argument_overloads{T <: RCPAssociative}(t::Type{T}) = [Trilinos.Teuchos.RCP{T}]

# Overload for size_t
@static if Sys.WORD_SIZE == 64
  CxxWrap.argument_overloads(t::Type{UInt64}) = [Int64]
end

include("Teuchos.jl")
include("Kokkos.jl")
include("Tpetra.jl")
include("Thyra.jl")


end # module
