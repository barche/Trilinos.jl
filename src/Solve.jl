@compat abstract type TrilinosSolver end

"""
Solver that works directly on Tpetra CrsMatrix and MultiVector, parametrized using a Teuchos ParameterList
"""
struct TpetraSolver{SolverT} <: TrilinosSolver
  solver::SolverT
  parameters::CxxWrap.SmartPointer{Teuchos.ParameterList}
  TpetraSolver{SolverT}(solver::SolverT,params::CxxWrap.SmartPointer{Teuchos.ParameterList}) where SolverT = new(solver,params)
end

function TpetraSolver(A::CxxUnion{Tpetra.CrsMatrix{ST,LT,GT,NT}}, parameterlist = Teuchos.ParameterList()) where {ST,LT,GT,NT}
  params = Teuchos.ParameterListPair(parameterlist, Teuchos.ParameterList())
  linear_solver_type = get(params, "Linear Solver Type", "Belos")
  solver_types = Teuchos.sublist(params, "Linear Solver Types")

  if linear_solver_type == "Belos"
    factory = Belos.SolverFactory{ST, Tpetra.MultiVector{ST,LT,GT,NT}, Tpetra.Operator{ST,LT,GT,NT}}()
    belos_parameters = Teuchos.sublist(solver_types, "Belos")
    solver_name = get(belos_parameters, "Solver Type", "BLOCK GMRES")
    belos_solver_types = Teuchos.sublist(belos_parameters, "Solver Types")
    solver_parameters = Teuchos.ParameterList(Teuchos.sublist(belos_solver_types.input, solver_name))

    solver = Belos.create(factory, solver_name, solver_parameters)
    linprob = Belos.LinearProblem(A)

    prec_type = get(params, "Preconditioner Type", "Ifpack2")
    if prec_type == "Ifpack2"
      prec_list = Teuchos.sublist(params, "Preconditioner Types")
      ifpack_list = Teuchos.sublist(prec_list, "Ifpack2")
      ifpack2_type = get(ifpack_list, "Preconditioner Type", "ILUT")
      ifpack2_params = Teuchos.sublist(ifpack_list, ifpack2_type)
      ifpack2_params_new = Teuchos.ParameterList(Teuchos.sublist(ifpack_list.input, ifpack2_type))
      prec_factory = Ifpack2.Factory()
      M = Ifpack2.create(prec_factory, ifpack2_type, A)
      Ifpack2.setParameters(M, ifpack2_params_new)
      Ifpack2.initialize(M)
      Ifpack2.compute(M)
      Belos.setRightPrec(linprob, M)
      ifpack_list.output[ifpack2_type] = ifpack2_params_new
    elseif prec_type == "MueLu"
      prec_list = Teuchos.sublist(params, "Preconditioner Types")
      muelu_list = Teuchos.sublist(prec_list, "MueLu")
      if isempty(muelu_list.input)
        MueLu.parameters_easy!(muelu_list.output)
      end

      muelu_prec = MueLu.CreateTpetraPreconditioner(A, muelu_list.output)
      Belos.setRightPrec(linprob, muelu_prec)
    elseif prec_type == "None"
      nothing
    else
      error("Unsupported  preconditioner type $prec_type")
    end

    Belos.setProblem(solver,linprob)

    belos_solver_types.output[solver_name] = solver_parameters
    return TpetraSolver{typeof(solver)}(solver, params.output)
  end
  error("Unsupported linear solver type: $linear_solver_type")
end

function default_parameters(solver_type="BLOCK GMRES"; prec_factory = Ifpack2.default_parameters)
  params = Dict("Linear Solver Type" => "Belos", "Linear Solver Types" => Dict("Belos" => Dict("Solver Type" => solver_type, "Solver Types" => Teuchos.ParameterList())))
  params["Preconditioner Type"] = "Ifpack2"

  prec_types = Teuchos.ParameterList("Preconditioner Types")
  prec_params = prec_factory()
  prec_types[Teuchos.name(prec_params)] = prec_params
  params["Preconditioner Types"] = prec_types

  stypes = params["Linear Solver Types"]["Belos"]["Solver Types"]
  sp = Teuchos.ParameterList()
  factory = Belos.SolverFactory{Float64, Tpetra.MultiVector{Float64,Int32,Int64,Kokkos.default_node_type()}, Tpetra.Operator{Float64,Int32,Int64,Kokkos.default_node_type()}}()
  solver = Belos.create(factory, solver_type, sp)
  stypes[solver_type] = sp
  finalize(solver)
  finalize(factory)
  return params
end

import Base: \

function \(A::TrilinosSolver, b::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}) where {ST,LT,GT,NT}
  prob = Belos.getProblem(A.solver)
  x = Tpetra.Vector(Tpetra.getDomainMap(Belos.getOperator(prob)))
  Belos.setProblem(prob,x,b)
  Belos.solve(A.solver)
  return x
end