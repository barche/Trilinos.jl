using Trilinos
using Test
using MPI
using LinearAlgebra

if Test.get_testset_depth() == 0
  MPI.Init()
  MPI.finalize_atexit()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

const n_mu = 10000
A = 20 * sparse(1.0I, n_mu, n_mu) + sprand(n_mu, n_mu, 20.0 / n_mu)
A = (A + A') / 2

rowmap = Tpetra.Map(n_mu, 0, comm)
A = Tpetra.CrsMatrix(A, rowmap)

muelu_prec = MueLu.CreateTpetraPreconditioner(A, MueLu.parameters_easy())
H = MueLu.GetHierarchy(muelu_prec)
@test MueLu.GetNumLevels(H) == 3

