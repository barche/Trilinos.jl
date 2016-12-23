# Powermethod using Tpetra
# Adapted from https://trilinos.org/docs/dev/packages/tpetra/doc/html/Tpetra_Lesson03.html

using MPI
using Trilinos

# MPI setup
MPI.Init()
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

println("Tpetra version is \"$(Tpetra.version())\"")

const num_global_indices = UInt(50)
const indexbase = 0
map = Tpetra.Map(num_global_indices, indexbase, comm)
@show typeof(map)

my_rank = Teuchos.getRank(comm)
num_my_elements = Tpetra.getNodeNumElements(map)

println("Number of elements for rank $my_rank is $num_my_elements")

MPI.Finalize()
