# Trilinos

This package aims to allow using the [Trilinos](https://trilinos.org/) library from Julia. The first objective is to expose the Belos solver library using the Tpetra matrix library, to allow solving linear systems using the modern methods currently developed in Trilinos. The current state should be considered as a proof-of-concept, implementing a 2D Laplace example as well as the powermethod from the [Tpetra tutorial](https://trilinos.org/docs/dev/packages/tpetra/doc/html/Tpetra_Lesson03.html).

At this time, the following is implemented (see tests and examples):
* Conversion from non-distributed Julia sparse array to Tpetra::CrsMatrix
* Vector views wrapping ArrayView (Julia loops as fast as in C++)
* Some native functions for matrix assembly (assembly loop on-par with C++, see laplace2d example)
* Linear solve using \
* Solver parametrization using ParameterList

## Installation

* Install the dependencies:
```julia
Pkg.add("MPI")
Pkg.add("CxxWrap")
Pkg.add("CustomUnitRanges")
```
* Set the environment variable `TRILINOS_ROOT` to the installation prefix of your Trilinos installation, e.g.
```julia
ENV["TRILINOS_ROOT"] = "/usr"
```
* Clone and build Trilinos.jl:
```julia
Pkg.clone("https://github.com/barche/Trilinos.jl")
Pkg.build("Trilinos")
```
