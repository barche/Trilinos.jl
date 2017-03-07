module Benchmark
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..RCPWrappable
  import ..RCPAssociative

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)
end
