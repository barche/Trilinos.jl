module Benchmark
  using CxxWrap, MPI
  import ..libjltrilinos

  @wrapmodule(libjltrilinos, :register_benchmark)
end
