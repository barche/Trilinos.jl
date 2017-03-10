module Testing
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..RCPWrappable
  import ..RCPAssociative
  import ..Teuchos

  registry = load_modules(_l_trilinos_wrap)
  wrap_module(registry)
end
