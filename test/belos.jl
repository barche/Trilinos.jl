using Trilinos
using Base.Test
using MPI

if Base.Test.get_testset_depth() == 0
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const n = 20
rowmap = Tpetra.Map(n, 0, comm)
A = Tpetra.CrsMatrix(spdiagm((ones(n),), (0,)),rowmap)

x = Tpetra.Vector(Tpetra.getDomainMap(A))
b = Tpetra.Vector(Tpetra.getRangeMap(A))
Tpetra.randomize(x)
Tpetra.randomize(b)

@show const OpT = supertype(supertype(supertype(typeof(A[]))))
@show const MVT = supertype(supertype(typeof(b[])))

solver_factory = Belos.SolverFactory{Float64, MVT, OpT}()

solparams = Teuchos.ParameterList()
solparams["Verbosity"] = Belos.StatusTestDetails + Belos.FinalSummary + Belos.TimingDetails
solver = Belos.create(solver_factory, "Block GMRES", solparams)
linprob = Belos.LinearProblem(A)

prec_factory = Ifpack2.Factory()
@show M = Ifpack2.create(prec_factory, "ILUT", A)
prec_params = Teuchos.ParameterList()
Ifpack2.setParameters(M, prec_params)
display(prec_params[])
Ifpack2.initialize(M)
Ifpack2.compute(M)

Belos.setRightPrec(linprob, M)
Belos.setProblem(linprob,x,b)
Belos.setProblem(solver, linprob)

@test Belos.solve(solver) == Belos.Converged
for (bi,xi) in zip(Tpetra.device_view(b),Tpetra.device_view(x))
  @test bi ≈ xi
end

println("Supported Belos solvers:")
foreach(println, Belos.supportedSolverNames(solver_factory))

if Base.Test.get_testset_depth() == 0
  MPI.Finalize()
end
