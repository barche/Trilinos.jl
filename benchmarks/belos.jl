using BenchmarkTools
using MPI
using Trilinos

MPI.Init()

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

function indefinite(n)
  # Generate an indefinite "hard" matrix
  srand(1)
  A = 4 * speye(n) + sprand(n, n, 60.0 / n)
  A = (A + A') / 2
  x = ones(n)
  b = A * x

  return A, b
end

# function posdef(n)
#   A, b = indefinite(n)
#   # Shift the spectrum a bit to make A positive definite
#   A += 2.35 * speye(n)

#   return A, b
# end

function posdef(n)
    A = spdiagm([fill(-1.0,n-1), fill(2.01, n), fill(-1.0, n-1)], (-1,0,1))
    b = A * ones(n)
    return A, b
end

function tpetra_system(S)
  A,b = S
  n = length(b)
  rowmap = Tpetra.Map(n, 0, comm)
  At = Tpetra.CrsMatrix(A,rowmap)
  bt = Tpetra.Vector(Tpetra.getRangeMap(At))

  bv = Tpetra.device_view(bt)
  for i in linearindices(b)
    bv[i-1] = b[i]
  end

  return At, bt
end

function gmres(; n = 10000, tol = 1e-5, restart::Int = 15, maxiter::Int = 1500)
  A, b = tpetra_system(indefinite(n))
  outer = div(maxiter, restart)

  println("Tolerance = ", tol, "; restart = ", restart, "; max #iterations = ", maxiter)

  params = Trilinos.default_parameters()
  solver_params = params["Linear Solver Types"]["Belos"]["Solver Types"]["Block GMRES"]
  solver_params["Convergence Tolerance"] = tol
  solver_params["Verbosity"] = reinterpret(Int32,Belos.StatusTestDetails)#reinterpret(Int32,Belos.FinalSummary) + reinterpret(Int32,Belos.TimingDetails)
  solver_params["Maximum Restarts"] = Int32(restart)
  solver_params["Block Size"] = Int32(maxiter)
  solver_params["Maximum Iterations"] = Int32(maxiter)
  
  solver = TpetraSolver(A, params)

  return @benchmark $solver \ $b
end

function cg(; n = 10000, tol = 1e-10, maxiter::Int = 1500)
  A, b = tpetra_system(posdef(n))
  @show typeof(A)

  println("Tolerance = ", tol, "; max #iterations = ", maxiter)

  params = Trilinos.default_parameters()
  solver_params = Teuchos.ParameterList()
  params["Linear Solver Types"]["Belos"]["Solver Type"] = "Block CG"
  solver_params["Convergence Tolerance"] = tol
  solver_params["Verbosity"] = reinterpret(Int32,Belos.StatusTestDetails)#reinterpret(Int32,Belos.FinalSummary) + reinterpret(Int32,Belos.TimingDetails)
  solver_params["Block Size"] = Int32(maxiter)
  solver_params["Maximum Iterations"] = Int32(maxiter)
  params["Linear Solver Types"]["Belos"]["Solver Types"]["Block CG"] = solver_params

  solver = TpetraSolver(A, params)

  return @benchmark $solver \ $b
end

gmres_t = gmres()
cg_t = cg(n=1_000_000, tol = 1e-6)

println("GMRES: $gmres_t")
println("CG: $cg_t")

MPI.Finalize()