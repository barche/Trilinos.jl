module Trilinos

export Belos
export Kokkos
export Teuchos
export Tpetra
export Thyra

using CxxWrap
using Compat

# Load the deps
const depsfile = joinpath(dirname(dirname(@__FILE__)), "deps", "deps.jl")
if !isfile(depsfile)
  error("$depsfile not found, package Trilinos did not build properly")
end
include(depsfile)

# Overload for size_t
@static if Sys.WORD_SIZE == 64
  CxxWrap.argument_overloads(t::Type{UInt64}) = [Int64]
end

@compat CxxUnion{T} = Union{T,CxxWrap.SmartPointer{T}}

include("Teuchos.jl")
include("Kokkos.jl")
include("Tpetra.jl")
include("Belos.jl")
include("Thyra.jl")
include("Benchmark.jl")
include("Testing.jl")

include("Solve.jl")

end # module
