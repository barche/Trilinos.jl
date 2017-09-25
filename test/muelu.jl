using Trilinos
using Base.Test
using MPI

if Base.Test.get_testset_depth() == 0
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

const n = 10000
A = 20 * speye(n) + sprand(n, n, 20.0 / n)
A = (A + A') / 2

rowmap = Tpetra.Map(n, 0, comm)
A = Tpetra.CrsMatrix(A, rowmap)

muelu_prec = MueLu.CreateTpetraPreconditioner(A, MueLu.parameters_easy())
H = MueLu.GetHierarchy(muelu_prec)
@test MueLu.GetNumLevels(H) == 3

if Base.Test.get_testset_depth() == 0
  MPI.Finalize()
end
