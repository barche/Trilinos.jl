using Trilinos
using Base.Test
using MPI

a = collect(1.0:10.0)

av = Teuchos.ArrayView(a)
@test Teuchos.size(av) == 10
