using Trilinos
using Base.Test
using MPI

if !MPI.Initialized()
  MPI.Init()
end

# The number of matrix rows
const n = 20

# Reuse the communicator from MPI.jl
comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
# Disribute n evenly over the number of processes
rowmap = Tpetra.Map(n, 0, comm)
# Construct a matrix from a Julia sparse unit matrix
A = Tpetra.CrsMatrix(spdiagm((ones(n),), (0,)),rowmap)

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

if !isdefined(:intesting)
  MPI.Finalize()
end
