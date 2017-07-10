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

pl3 = Teuchos.ParameterList()
pl2["Manual Sublist"] = pl3
@test Teuchos.name(pl2["Manual Sublist"]) == "ANONYMOUS"

@test Teuchos.isSublist(pl, "child")
@test pl["child"]["param1"] == "test"

pl2_ref = pl["child"]
pl2_ref["param1"] = "test2"

@test pl["child"]["param1"] == "test2"

@test get(pl, "intparam", 0) == 2
@test get(pl, "thisdoesnotexist", "somedefault") == "somedefault"

pl_copy = Teuchos.ParameterList(pl)
pl_copy["intparam"] = 3
@test pl["intparam"] == 2
@test pl_copy["intparam"] == 3
pl_copy["child"]["param1"] = "copy_modified"
@test pl["child"]["param1"] == "test2"
@test pl_copy["child"]["param1"] == "copy_modified"

pl["enum"] = Teuchos.VERB_DEFAULT
@test pl["enum"] == -1

display(pl[])
println()
