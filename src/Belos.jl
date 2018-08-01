module Belos
  using CxxWrap, MPI
  import ..libjltrilinos
  import ..Teuchos

  @wrapmodule(libjltrilinos, :register_belos)
end
