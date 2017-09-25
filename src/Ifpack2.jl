module Ifpack2
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..Teuchos
  import ..Tpetra

  wrap_module(_l_trilinos_wrap, Ifpack2)

  function default_parameters(ifpack_prec = "ILUT")
    pl = Teuchos.ParameterList("Ifpack2")
    ifpack2 = Teuchos.ParameterList("Ifpack2")
    pl["Preconditioner Type"] = ifpack_prec
    pl[ifpack_prec] = Teuchos.ParameterList(ifpack_prec)
  end

end
