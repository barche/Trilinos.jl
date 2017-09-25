using BenchmarkTools
using MPI
using Trilinos
using Base.Test

MPI.Init()

#Kokkos.initialize(Kokkos.OpenMP(), 4)

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

my_rank = Teuchos.getRank(comm)
if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

function indefinite(n)
  # Generate an indefinite "hard" matrix
  srand(1)
  A = 20 * speye(n) + sprand(n, n, 20.0 / n)
  A = (A + A') / 2
  x = ones(n)
  b = A * x

  return A, b, ones(n)
end


# The following Laplace 2D implementation is based on:
# https://github.com/lruthotto/KrylovMethods.jl/blob/master/benchmarks/benchmark2DLaplacian.ipynb

"""
ibc,iin = getBoundaryIndices(n::Int)

returns indices of boundary and interior nodes of regular mesh with n^2 cells.
"""
function getBoundaryIndices(n::Int)
    ids = reshape(collect(1:(n+1)^2),n+1,n+1)
    iin = vec(ids[2:end-1,2:end-1])
    ibc = setdiff(vec(ids),vec(iin))
    return ibc,iin
end

"""
L,Lin,Lib = getLaplacian(n::Int)

returns discrete Laplacian, interior part, and boundary part, on regular mesh with n^2 cells.
"""
function getLaplacian(n::Int)
    h  = 1/n
    dx = spdiagm((-ones(n,1),ones(n,1)),0:1,n,n+1)/h
    d2x = dx'*dx
    L  = kron(speye(n+1),d2x) + kron(d2x,speye(n+1))
    
    # split into boundary and interior part
    ibc,iin = getBoundaryIndices(n)
    Lin = L[iin,iin]
    Lib = L[iin,ibc]
    return L,Lin,Lib
end
"""
g = getBoundaryConditions(n::Int,fctn=(x,y)->5.*y*sin(pi*x))

discretizes function on regular grid with n^2 cells and returns boundary and interior values as well as the source term value
"""
function getBoundaryConditions(n::Int,dirichlet = (x,y) -> (1-x^2)*(1-y^2), sourceterm = (x, y) -> 2*((1-x^2)+(1-y^2)))
    x  = linspace(0,1,n+1)
    g = map(Float64,[dirichlet(x1,x2) for x1 in x, x2 in x]')
    s = map(Float64,[sourceterm(x1,x2) for x1 in x, x2 in x]')
    ibc,iin = getBoundaryIndices(n)
    return vec(g)[ibc], vec(g)[iin], vec(s)[iin]
end

function laplace2d(n=1000)
  L,Lin,Lib = getLaplacian(n)
  ubc,refsol,f      = getBoundaryConditions(n)
  return Lin,vec(-Lib*ubc)+f,refsol
end

function tpetra_system(S)
  A,b,x = S
  n = length(b)
  rowmap = Tpetra.Map(n, 0, comm) #, Kokkos.KokkosOpenMPWrapperNode)
  At = Tpetra.CrsMatrix(A,rowmap)

  rmap = Tpetra.getRangeMap(At)
  bt = Tpetra.Vector(rmap)
  xt = Tpetra.Vector(rmap)

  num_my_elements = Tpetra.getNodeNumElements(rowmap)

  bv = Tpetra.device_view(bt)
  xv = Tpetra.device_view(xt)
  for local_row = 0:num_my_elements-1
    global_row = Tpetra.getGlobalElement(rmap, local_row)
    bv[local_row] = b[global_row+1]
    xv[local_row] = x[global_row+1]
  end

  return At, bt, xt
end

"""
Builds a Belos / Tpetra linear system, solving it once and checking the result against the reference solution
"""
function build_checked_system(f, solver_name, prec_type, n, tol, restart::Int, maxiter::Int)
  A, b, refsol = tpetra_system(f(n))

  println("Tolerance = ", tol, "; restart = ", restart, "; max #iterations = ", maxiter)

  params = Trilinos.default_parameters(solver_name)
  solver_params = params["Linear Solver Types"]["Belos"]["Solver Types"][solver_name]
  solver_params["Convergence Tolerance"] = tol
  solver_params["Verbosity"] = Belos.StatusTestDetails + Belos.FinalSummary + Belos.TimingDetails
  solver_params["Maximum Restarts"] = Int32(restart)
  solver_params["Block Size"] = Int32(maxiter)
  solver_params["Maximum Iterations"] = Int32(maxiter)
  params["Preconditioner Type"] = prec_type
  
  solver = TpetraSolver(A, params)

  sol = solver \ b
  v = Tpetra.device_view(sol)
  refv = Tpetra.device_view(refsol)
  maxerr = 0
  for i in linearindices(v)
    maxerr = max(abs(v[i] - refv[i]), maxerr)
  end
  @test maxerr < 100*tol

  return solver, b
end

sol_ndef(solver_name; n = 500000, tol = 1e-5, restart::Int = 15, maxiter::Int = 1500) = build_checked_system(indefinite, solver_name, "None", n, tol, restart, maxiter)

"""
Note: n is not the size of the matrix, but n in an n × n grid!
"""
sol_posdef(solver_name; n = 1000, tol = 1e-10, maxiter::Int = 1500) = build_checked_system(laplace2d, solver_name, "Ifpack2", n, tol, 0, maxiter)

suite = BenchmarkGroup()

suite["GMRES"] = @benchmarkable x=A\b setup=((A,b) = sol_ndef("GMRES"))
suite["BICGSTAB"] = @benchmarkable x=A\b setup=((A,b) = sol_ndef("BICGSTAB"))
suite["CG"] = @benchmarkable A\b setup=((A,b) = sol_posdef("CG",n=300))

result = run(suite)
if(my_rank == 0)
  display(result)
end

MPI.Finalize()