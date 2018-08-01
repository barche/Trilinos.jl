using MPI
using Test

MPI.Init()
MPI.finalize_atexit()

excluded = ["runtests.jl"]

@testset "Trilinos tests" begin
  @testset "$f" for f in filter(fname -> fname ∉ excluded, readdir())
    println("======== Starting tests from $f ========")
    include(f)
  end
  sleep(1)
end
