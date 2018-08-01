using Trilinos
using Test
using MPI

if Test.get_testset_depth() == 0
  MPI.Init()
end

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))

const n = 20

# tri-diagonal nxn test matrix
A = spdiagm(-1 => 2*ones(n-1), 0 => ones(n), 1=> 3*ones(n-1))

rowmap = Tpetra.Map(n, 0, comm)
A_crs = Tpetra.CrsMatrix(A,rowmap)
Tpetra.describe(A_crs, Teuchos.VERB_LOW)

@test sqrt(sum(nonzeros(A).^2)) ≈ Tpetra.getFrobeniusNorm(A_crs)

function loop_gid(rowmap)
  n_my_elms = Tpetra.getNodeNumElements(rowmap)
  i_sum  = 0
  for i in 1:n_my_elms
    i_sum += Tpetra.getGlobalElement(rowmap,i)
  end
  return i_sum
end

@show methods(Tpetra.getGlobalElement)

@time loop_gid(rowmap[])
@time loop_gid(rowmap[])
@time loop_gid(rowmap[])

if Test.get_testset_depth() == 0
  MPI.Finalize()
end
