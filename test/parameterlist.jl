using Trilinos
using Base.Test
using MPI

pl = Teuchos.ParameterList("test1")
@test Teuchos.name(pl) == "test1"

Teuchos.setName(pl, "test2")
@test Teuchos.name(pl) == "test2"

@test Teuchos.numParams(pl) == 0
