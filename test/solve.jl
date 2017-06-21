using Trilinos
using Base.Test
using MPI

if !MPI.Initialized()
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const n = 20
rowmap = Tpetra.Map(n, 0, comm)
A = Tpetra.CrsMatrix(spdiagm((ones(n),), (0,)),rowmap)

solver = TpetraSolver(A)
b = Tpetra.Vector(Tpetra.getRangeMap(A))
Tpetra.randomize(b)

x = solver \ b

# Check result
bv = Tpetra.device_view(b)
xv = Tpetra.device_view(x)
for (bi,xi) in zip(bv,xv)
  @test bi ≈ xi
end

if !isdefined(:intesting)
  MPI.Finalize()
end
