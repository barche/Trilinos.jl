module MueLu
  using CxxWrap, MPI
  import .._l_trilinos_wrap
  import ..Teuchos
  import ..Tpetra

  wrap_module(_l_trilinos_wrap, MueLu)

  """
  Return the "easy" parameters from the MueLu tutorial
  """
  function parameters_easy()
    pl = Teuchos.ParameterList("MueLu")
    pl["verbosity"] = "low"
    pl["max levels"] = 3
    pl["coarse: max size"] = 10
    pl["multigrid algorithm"] = "sa"

    pl["smoother: type"] = "RELAXATION"
    smoother_params = Teuchos.sublist(pl, "smoother: params")
    smoother_params["relaxation: type"] = "Jacobi"
    smoother_params["relaxation: sweeps"] = 1
    smoother_params["relaxation: damping factor"] = 0.9

    pl["aggregation: type"] = "uncoupled"
    pl["aggregation: min agg size"] = 3
    pl["aggregation: max agg size"] = 9

    return pl
  end

end
