using Trilinos
using Base.Test
using MPI

if Base.Test.get_testset_depth() == 0
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

@show device_mvt = Tpetra.device_view_type(mv)
@show Kokkos.value_type(device_mvt)
@show Kokkos.array_layout(device_mvt)
@show host_mvt = Tpetra.host_view_type(mv)

host_v = Tpetra.getLocalView(host_vt, v)
host_mv = Tpetra.getLocalView(host_mvt, mv)
@test Kokkos.dimension(host_v, 0) == num_my_elements
@test Int(Kokkos.dimension(host_v, 1)) == 1
@test unsafe_load(host_v(0,0)) == 1.0
@test unsafe_load(host_mv(0,2)) == 2.0

# high-level interface, views are 0-based for consistency with the Trilinos API
hv1 = Tpetra.host_view(v)
@test length(linearindices(hv1)) == num_my_elements
if my_rank == 0
  hv1[0] = 3
end
if my_rank == (Teuchos.getSize(comm)-1)
  hv1[last(indices(hv1,1))] = 4
end
@show hv1
@test Tpetra.dot(v,v) == (n-2)+9.0+16.0

dv2 = Tpetra.device_view(mv)
@test map(length,indices(dv2)) == (num_my_elements, mv_cols)
if my_rank == 0
  dv2[0,0] = 3
end
if my_rank == (Teuchos.getSize(comm)-1)
  dv2[last(indices(dv2,1)),2] = 4
end

dots_check = [(n-1)*4.0+9.0, n*4.0, (n-1)*4.0+16.0]
mv_dots = zeros(mv_cols)
Tpetra.dot(mv,mv,Teuchos.ArrayView(mv_dots))
@test mv_dots == dots_check

if my_rank == 0
  println("hv1:\n",hv1)
  println("dv2:\n",dv2)
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
  for i in linearindices(av)
    av[i] = i
  end
end

function benchmark_fill_abstractarray2(rowmap, a)
  num_my_elements = Int(Tpetra.getNodeNumElements(rowmap))
  av = Tpetra.device_view(a)
  for j in indices(av,2)
    for i in indices(av,1)
      av[i,j] = i*(j+1)
    end
  end
end

const bench_size = 1000000
benchmap = Tpetra.Map(bench_size, 0, comm)
benchvec = Tpetra.Vector(benchmap)
benchview = Tpetra.device_view(benchvec)
n_my_elms = Int(Tpetra.getNodeNumElements(benchmap))
v_end = n_my_elms-1

Tpetra.putScalar(benchvec, 5.0)
println("lowlevel timings:")
@time benchmark_fill_lowlevel(benchmap, benchvec)
@time benchmark_fill_lowlevel(benchmap, benchvec)
@time benchmark_fill_lowlevel(benchmap, benchvec)

@test benchview[0] == 0
@test benchview[v_end] == n_my_elms-1

Tpetra.putScalar(benchvec, 5.0)
println("abstractarray timings:")
@time benchmark_fill_abstractarray(benchmap, benchvec)
@time benchmark_fill_abstractarray(benchmap, benchvec)
@time benchmark_fill_abstractarray(benchmap, benchvec)

@test benchview[0] == 0
@test benchview[v_end] == n_my_elms-1

Tpetra.putScalar(benchvec, 5.0)
println("C++ timings:")
@time Trilinos.Benchmark.vector_fill(benchmap, benchvec)
@time Trilinos.Benchmark.vector_fill(benchmap, benchvec)
@time Trilinos.Benchmark.vector_fill(benchmap, benchvec)

@test benchview[0] == 0
@test benchview[v_end] == n_my_elms-1

benchmv = Tpetra.MultiVector(benchmap, mv_cols)
benchmvview = Tpetra.device_view(benchmv)

Tpetra.putScalar(benchmv, 5.0)
println("abstractarray timings, MultiVector with linear indexing:")
@time benchmark_fill_abstractarray(benchmap, benchmv)
@time benchmark_fill_abstractarray(benchmap, benchmv)
@time benchmark_fill_abstractarray(benchmap, benchmv)

@test benchmvview[0,0] == 1
@test benchmvview[v_end,mv_cols-1] == n_my_elms*mv_cols

Tpetra.putScalar(benchmv, 5.0)
println("abstractarray timings, MultiVector with [i,j] indexing:")
@time benchmark_fill_abstractarray2(benchmap, benchmv)
@time benchmark_fill_abstractarray2(benchmap, benchmv)
@time benchmark_fill_abstractarray2(benchmap, benchmv)

@test benchmvview[0,0] == 0
@test benchmvview[v_end,mv_cols-1] == (n_my_elms-1)*mv_cols

Tpetra.putScalar(benchmv, 5.0)
println("C++ timings, MultiVector:")
@time Trilinos.Benchmark.multivector_fill(benchmap, benchmv)
@time Trilinos.Benchmark.multivector_fill(benchmap, benchmv)
@time Trilinos.Benchmark.multivector_fill(benchmap, benchmv)

@test benchmvview[0,0] == 0
@test benchmvview[v_end,mv_cols-1] == (n_my_elms-1)*mv_cols

if Base.Test.get_testset_depth() == 0
  MPI.Finalize()
end
