using Trilinos
using Test
using MPI

if Test.get_testset_depth() == 0
  MPI.Init()
end

ccomm = MPI.CComm(MPI.COMM_WORLD)
comm = Teuchos.MpiComm(ccomm)

@test Teuchos.getRank(comm) == MPI.Comm_rank(MPI.COMM_WORLD)
@test Teuchos.getSize(comm) == MPI.Comm_size(MPI.COMM_WORLD)

if Test.get_testset_depth() == 0
  MPI.Finalize()
end
