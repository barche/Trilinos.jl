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

@show device_vt = Tpetra.device_view_type(v)
@show host_vt = Tpetra.host_view_type(v)

@show typeof(device_vt)
@show typeof(host_vt)

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

function benchmark_fill_lowlevel(rowmap, a)
  num_my_elements = Int(Tpetra.getNodeNumElements(rowmap))
  av = Tpetra.getLocalView(Tpetra.device_view_type(a), a)
  for i in 0:(num_my_elements-1)
    unsafe_store!(av(i,0), i)
  end
end

function benchmark_fill_abstractarray(rowmap, a)
  num_my_elements = Int(Tpetra.getNodeNumElements(rowmap))
  av = Tpetra.device_view(a)
  for i in 1:num_my_elements
    @inbounds av[i] = i
  end
end

const bench_size = 1000000
benchmap = Tpetra.Map(bench_size, 0, comm)
benchvec = Tpetra.Vector(benchmap)
benchview = Tpetra.device_view(benchvec)
n_my_elms = Int(Tpetra.getNodeNumElements(benchmap))

Tpetra.putScalar(benchvec, 5.0)
println("lowlevel timings:")
@time benchmark_fill_lowlevel(benchmap, benchvec)
@time benchmark_fill_lowlevel(benchmap, benchvec)
@time benchmark_fill_lowlevel(benchmap, benchvec)

@test benchview[1] == 0
@test benchview[end] == n_my_elms-1

Tpetra.putScalar(benchvec, 5.0)
println("abstractarray timings:")
@time benchmark_fill_abstractarray(benchmap, benchvec)
@time benchmark_fill_abstractarray(benchmap, benchvec)
@time benchmark_fill_abstractarray(benchmap, benchvec)

@test benchview[1] == 1
@test benchview[end] == n_my_elms

Tpetra.putScalar(benchvec, 5.0)
println("C++ timings:")
@time Trilinos.Benchmark.vector_fill(benchmap, benchvec)
@time Trilinos.Benchmark.vector_fill(benchmap, benchvec)
@time Trilinos.Benchmark.vector_fill(benchmap, benchvec)

@test benchview[1] == 0
@test benchview[end] == n_my_elms-1

if !isdefined(:intesting)
  MPI.Finalize()
end
