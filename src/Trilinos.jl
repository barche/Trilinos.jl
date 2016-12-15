module Trilinos

export Teuchos

using CxxWrap
using MPI

const depsfile = joinpath(dirname(dirname(@__FILE__)), "deps", "deps.jl")
if !isfile(depsfile)
  error("$depsfile not found, package Trilinos did not build properly")
end
include(depsfile)

wrap_modules(_l_trilinos_wrap)

end # module
