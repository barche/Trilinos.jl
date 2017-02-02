using Trilinos
using Base.Test
using MPI

pl = Teuchos.ParameterList("test1")
@test Teuchos.name(pl) == "test1"

Teuchos.setName(pl, "test2")
@test Teuchos.name(pl) == "test2"

@test Teuchos.numParams(pl) == 0

Teuchos.set(pl, "intparam", 1)
@test Teuchos.isParameter(pl, "intparam")
@test !Teuchos.isParameter(pl, "dummy")
@test Teuchos.isType(Int, pl, "intparam")
@test !Teuchos.isType(Int, pl, "dummy")
@test_throws ErrorException Teuchos.get(Int, pl, "dummy")
@test Teuchos.get(Int, pl, "intparam") == 1
@test Teuchos.get_type(pl, "intparam") == Int
@test pl["intparam"] == 1
pl["intparam"] = 2
@test pl["intparam"] == 2

pl["dblparam"] = 2.0
@test length(pl) == 2
@test pl["dblparam"] == 2.0

@test_throws KeyError pl["dummy"]

pl2 = Teuchos.sublist(pl, "child")
pl2["param1"] = "test"

@test Teuchos.isSublist(pl, "child")
@test pl["child"]["param1"] == "test"

display(pl[])
println()
