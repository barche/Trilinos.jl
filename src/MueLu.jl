module MueLu
  using CxxWrap, MPI
  import ..libjltrilinos
  import ..Teuchos
  import ..Tpetra

  @wrapmodule(libjltrilinos, :register_muelu)

  """
  Return the "easy" parameters from the MueLu tutorial
  """
  function parameters_easy()
    pl = Teuchos.ParameterList("MueLu")
    parameters_easy!(pl)
    return pl
  end
  function parameters_easy!(pl)
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

    return
  end

end
