module Ifpack2
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..Teuchos
  import ..Tpetra

  wrap_module(_l_trilinos_wrap, Ifpack2)
end
