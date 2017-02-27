using Trilinos
using Base.Test
using MPI

if !MPI.Initialized()
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

const n = 20

# tri-diagonal nxn test matrix
A = spdiagm((2*ones(n-1),ones(n),3*ones(n-1)), (-1,0,1))

rowmap = Tpetra.Map(n, 0, comm)
A_crs = Tpetra.CrsMatrix(A,rowmap)

@test sqrt(sum(nonzeros(A).^2)) ≈ Tpetra.getFrobeniusNorm(A_crs)

if !isdefined(:intesting)
  MPI.Finalize()
end
