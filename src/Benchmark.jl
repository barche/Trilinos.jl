module Benchmark
  using CxxWrap, MPI
  import .._l_trilinos_wrap

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)
end
