module Testing
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..Teuchos

  wrap_module(_l_trilinos_wrap, Testing)
end
