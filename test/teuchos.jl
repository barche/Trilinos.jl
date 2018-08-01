using Trilinos
using Test

L = 10
teuchosarray = Teuchos.Array(L, "test")
@test size(teuchosarray) == (L,)
@test length(teuchosarray) == L
@test axes(teuchosarray) == (1:L,)

for (i,str) in enumerate(teuchosarray)
  teuchosarray[i] *= string(i)
end

println(teuchosarray)

for (i,str) in enumerate(teuchosarray)
  @test teuchosarray[i] == "test$i"
end

@test convert(Int32,Teuchos.null) == 0