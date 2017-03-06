using Trilinos
using Base.Test
using MPI

if !MPI.Initialized()
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const n = 20
rowmap = Tpetra.Map(n, 0, comm)
num_my_elements = Int(Tpetra.getNodeNumElements(rowmap))

my_rank = Teuchos.getRank(comm)
if my_rank != 0
  redirect_stdout(open("/dev/null", "w"))
  redirect_stderr(open("/dev/null", "w"))
end

# Number of columns for the multivector
mv_cols = 3

v = Tpetra.Vector(rowmap)
mv = Tpetra.MultiVector(rowmap, mv_cols)

Tpetra.putScalar(v, 1)
Tpetra.putScalar(mv, 2)

device_vt = Tpetra.device_view_type(v)
host_vt = Tpetra.host_view_type(v)

device_mvt = Tpetra.device_view_type(mv)
host_mvt = Tpetra.host_view_type(mv)

host_v = Tpetra.getLocalView(host_vt, v)
host_mv = Tpetra.getLocalView(host_mvt, mv)
@test Kokkos.dimension(host_v, 0) == num_my_elements
@test Int(Kokkos.dimension(host_v, 1)) == 1
@test unsafe_load(host_v(0,0)) == 1.0
@test unsafe_load(host_mv(0,2)) == 2.0

# high-level interface
hv1 = Tpetra.host_view(v)
@test size(hv1) == (num_my_elements,)
if my_rank == 0
  hv1[1] = 3
end
if my_rank == (Teuchos.getSize(comm)-1)
  hv1[end] = 4
end
@test Tpetra.dot(v,v) == (n-2)+9.0+16.0

dv2 = Tpetra.device_view(mv)
@test size(dv2) == (num_my_elements, mv_cols)
if my_rank == 0
  dv2[1,1] = 3
end
if my_rank == (Teuchos.getSize(comm)-1)
  dv2[end,3] = 4
end

dots_check = [(n-1)*4.0+9.0, n*4.0, (n-1)*4.0+16.0]
mv_dots = zeros(mv_cols)
Tpetra.dot(mv,mv,Teuchos.ArrayView(mv_dots))
@test mv_dots == dots_check

if my_rank == 0
  display(hv1)
  println()
  display(dv2)
  println()
end


if !isdefined(:intesting)
  MPI.Finalize()
end
