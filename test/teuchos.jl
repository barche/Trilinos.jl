using Trilinos
using Base.Test

L = 10
teuchosarray = Teuchos.Array(L, "test")
@test size(teuchosarray) == (L,)
@test length(teuchosarray) == L
@test indices(teuchosarray) == (1:L,)

for (i,s) in enumerate(teuchosarray)
  teuchosarray[i] *= string(i)
end

println(teuchosarray)

for (i,s) in enumerate(teuchosarray)
  @test teuchosarray[i] == "test$i"
end

@test convert(Int32,Teuchos.null) == 0