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
    solver_name = get(params, "Solver Type", "Block GMRES")
    belos_solver_types = Teuchos.sublist(belos_parameters, "Solver Types")
    solver_parameters = Teuchos.ParameterList(Teuchos.sublist(belos_solver_types.input, solver_name))

    solver = Belos.create(factory, solver_name, solver_parameters)
    linprob = Belos.LinearProblem(A)
    Belos.setProblem(solver,linprob)

    belos_solver_types.output[solver_name] = solver_parameters
    return TpetraSolver{typeof(solver)}(solver, params.output)
  end
  error("Unsupported linear solver type: $linear_solver_type")
end

import Base: \

function \(A::TrilinosSolver, b::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}}) where {ST,LT,GT,NT}
  prob = Belos.getProblem(A.solver)
  x = Tpetra.Vector(Tpetra.getDomainMap(Belos.getOperator(prob)))
  Belos.setProblem(prob,x,b)
  Belos.solve(A.solver)
  return x
end