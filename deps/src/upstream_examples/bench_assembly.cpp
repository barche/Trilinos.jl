#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_VerboseObject.hpp>


Teuchos::RCP<Teuchos::Time> graph_time;
Teuchos::RCP<Teuchos::Time> fill_time;

typedef double scalar_t;
typedef int local_t;
typedef int64_t global_t;


template<typename GraphT>
void build_graph(GraphT& A)
{
  Teuchos::TimeMonitor local_timer(*graph_time);
  const auto& rowmap = *A.getRowMap();
  local_t n_my_elms = rowmap.getNodeNumElements();
  const global_t gmin = rowmap.getMinGlobalIndex();
  const global_t gmax = rowmap.getMaxGlobalIndex();

  // storage for the per-row values
  global_t row_indices_arr[5] = {0,0,0,0,0};
  Teuchos::ArrayView<global_t> row_indices(row_indices_arr,5);

  for(local_t i = 0; i != n_my_elms; ++i)
  {
    const global_t global_row =  rowmap.getGlobalElement(i);
    for(int j = 0; j != 5; ++j)
    {
      row_indices[j] = global_row - (2-j);
      if(row_indices[j] < gmin || row_indices[j] > gmax)
      {
        row_indices[j] = global_row;
      }
    }
    A.insertGlobalIndices(global_row, row_indices);
  }
}

template<typename MatrixT>
void fill_matrix(MatrixT& A)
{
  Teuchos::TimeMonitor local_timer(*fill_time);
  const auto& rowmap = *A.getRowMap();
  const local_t n_my_elms = rowmap.getNodeNumElements();

  // storage for the per-row values
  local_t row_indices_arr[5] = {0,0,0,0,0};
  scalar_t row_values_arr[5] = {0,0,0,0,0};

  Teuchos::ArrayView<local_t> row_indices(row_indices_arr,5);
  Teuchos::ArrayView<scalar_t> row_values(row_values_arr,5);
  std::size_t n_elems;
  for(local_t row = 0; row != n_my_elms; ++row)
  {
    A.getLocalRowCopy(row, row_indices, row_values, n_elems);
    for (std::size_t j = 0; j != n_elems; ++j)
    {
      row_values[j] = static_cast<scalar_t>(j+1);
    }
    A.replaceLocalValues(row, row_indices, row_values);
  }
}


template<typename comm_t>
void bench_assembly(comm_t comm, const global_t nnodes)
{
  auto map = Teuchos::rcp(new Tpetra::Map<local_t,global_t>(nnodes, 0, comm));
  typedef Tpetra::Map<local_t,global_t>::node_type node_t;

  auto matrix_graph = Teuchos::rcp(new Tpetra::CrsGraph<local_t,global_t>(map, 5));
  build_graph(*matrix_graph);
  matrix_graph->fillComplete();

  // Matrix construction and fill
  auto A = Teuchos::rcp(new Tpetra::CrsMatrix<scalar_t,local_t,global_t>(matrix_graph));
  A->resumeFill();
  fill_matrix(*A);
  A->fillComplete();

  //A->describe(*Teuchos::VerboseObjectBase::getDefaultOStream(), Teuchos::VERB_EXTREME);
}

void do_run()
{
  graph_time = Teuchos::TimeMonitor::getNewCounter("Graph construction time");
  fill_time = Teuchos::TimeMonitor::getNewCounter("Matrix fill time");

  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  const global_t N = 1000000;
  
  for(int i = 0; i != 3; ++i)
  {
    bench_assembly(comm, N);
    Teuchos::TimeMonitor::report(std::cout);
    Teuchos::TimeMonitor::zeroOutTimers();
  }
}

int main (int argc, char *argv[])
{
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);
  
  do_run();

  return 0;
}
