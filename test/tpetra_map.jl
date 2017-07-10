using Trilinos
using Base.Test
using MPI

if Base.Test.get_testset_depth() == 0
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

const n = 20

rowmap0 = Tpetra.Map(n, 0, comm)
@test Tpetra.getIndexBase(rowmap0) == 0

rowmap1 = Tpetra.Map(n, 1, comm)
@test Tpetra.getIndexBase(rowmap1) == 1

if Base.Test.get_testset_depth() == 0
  MPI.Finalize()
end
