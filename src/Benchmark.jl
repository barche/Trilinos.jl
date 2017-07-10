module Benchmark
  using CxxWrap, MPI
  import .._l_trilinos_wrap

  wrap_module(_l_trilinos_wrap, Benchmark)
end
