using Trilinos
using Base.Test
using MPI

if !MPI.Initialized()
  MPI.Init()
end

ccomm = MPI.CComm(MPI.COMM_WORLD)
comm = Teuchos.MpiComm(ccomm)

@test Teuchos.getRank(comm) == MPI.Comm_rank(MPI.COMM_WORLD)
@test Teuchos.getSize(comm) == MPI.Comm_size(MPI.COMM_WORLD)

if !isdefined(:intesting)
  MPI.Finalize()
end
