using MPI
using Base.Test

MPI.Init()

excluded = ["runtests.jl"]

@testset "Trilinos tests" begin
  @testset "$f" for f in filter(fname -> fname ∉ excluded, readdir())
    include(f)
  end
  sleep(1)
end

MPI.Finalize()
