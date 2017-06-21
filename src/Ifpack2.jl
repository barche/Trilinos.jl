module Ifpack2
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..Teuchos
  import ..Tpetra

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)
end
