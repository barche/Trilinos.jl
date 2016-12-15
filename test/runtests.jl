using Trilinos
using Base.Test
using MPI

MPI.Init()
ccomm = MPI.CComm(MPI.COMM_WORLD)

comm = Teuchos.MpiComm(ccomm)

@test Teuchos.getrank(comm) == MPI.Comm_rank(MPI.COMM_WORLD)
@test Teuchos.getsize(comm) == MPI.Comm_size(MPI.COMM_WORLD)
