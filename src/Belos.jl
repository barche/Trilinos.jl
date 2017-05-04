module Belos
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..Teuchos

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)
end
