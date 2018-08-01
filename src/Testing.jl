module Testing
  using CxxWrap, MPI
  import ..libjltrilinos
  import ..Teuchos

  @wrapmodule(libjltrilinos, :register_testing)
end
