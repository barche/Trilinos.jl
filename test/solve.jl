using Trilinos
using Test
using MPI

if Test.get_testset_depth() == 0
  MPI.Init()
end

# The number of matrix rows
const n_sol = 20

# Reuse the communicator from MPI.jl
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
# Disribute n_sol evenly over the number of processes
rowmap = Tpetra.Map(n_sol, 0, comm)
# Construct a matrix from a Julia sparse unit matrix
A = Tpetra.CrsMatrix(spdiagm(0 => ones(n_sol)),rowmap)

# Solver with default parameter
solver = TpetraSolver(A)
# Random RHS
b = Tpetra.Vector(Tpetra.getRangeMap(A))
Tpetra.randomize(b)

# solve the system
x = solver \ b

# Get a local vector view
bv = Tpetra.device_view(b)
xv = Tpetra.device_view(x)
# This loop over raw data is as fast as in C++
for (bi,xi) in zip(bv,xv)
  @test bi ≈ xi
end

if Test.get_testset_depth() == 0
  MPI.Finalize()
end
