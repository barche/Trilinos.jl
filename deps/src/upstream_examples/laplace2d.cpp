#include <Thyra_BelosLinearOpWithSolveFactory.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
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


Teuchos::RCP<Teuchos::Time> assembly_time;

//#include <Galeri_XpetraMatrixTypes.hpp>

typedef double scalar_t;
typedef int local_t;
typedef int64_t global_t;

local_t laplace2d_indices(global_t inds_array[5], const global_t i, const global_t nx, const global_t ny)
{
  const global_t ix = i % nx;
  const global_t iy = (i-ix) / nx;

  local_t n_inds = 0;
  inds_array[n_inds] = i;
  if(iy != ny-1)
  {
      n_inds += 1;
      inds_array[n_inds] = i+nx;
  }
  if(ix != 0)
  {
      n_inds += 1;
      inds_array[n_inds] = i-1;
  }
  if(iy != 0)
  {
      n_inds += 1;
      inds_array[n_inds] = i-nx;
  }
  if(ix != nx-1)
  {
      n_inds += 1;
      inds_array[n_inds] = i+1;
  }
  return n_inds+1;
}

template<typename MatrixT>
void fill_laplace2d(MatrixT& A, const global_t nx, const global_t ny)
{
  Teuchos::TimeMonitor local_timer(*assembly_time);
  const auto& rowmap = *A.getRowMap();
  local_t n_my_elms = rowmap.getNodeNumElements();

  // storage for the per-row values
  global_t row_indices[5] = {0,0,0,0,0};
  scalar_t row_values[5] = {4.0,-1.0,-1.0,-1.0,-1.0};

  for(local_t i = 0; i != n_my_elms; ++i)
  {
    global_t global_row = rowmap.getGlobalElement(i);
    local_t row_n_elems = laplace2d_indices(row_indices, global_row, nx, ny);
    row_values[0] = 4.0 - (5-row_n_elems);
    A.insertGlobalValues(global_row, Teuchos::ArrayView<global_t>(row_indices,row_n_elems), Teuchos::ArrayView<scalar_t>(row_values,row_n_elems));
  }
}

template<typename comm_t>
void laplace2d(comm_t comm, const global_t nx, const global_t ny)
{
  auto map = Teuchos::rcp(new Tpetra::Map<local_t,global_t>(nx*ny, 0, comm));
  typedef Tpetra::Map<local_t,global_t>::node_type node_t;

  // Matrix construction and fill
  auto A = Teuchos::rcp(new Tpetra::CrsMatrix<scalar_t,local_t,global_t>(map, 0));
  fill_laplace2d(*A, nx, ny);
  A->fillComplete();

  //A->describe(*Teuchos::VerboseObjectBase::getDefaultOStream(), Teuchos::VERB_EXTREME);

  //-------------------------- Check with Galeri assembly
  // auto A_galeri = Galeri::Xpetra::Cross2D<scalar_t, local_t, global_t, Tpetra::Map<local_t,global_t>, Tpetra::CrsMatrix<scalar_t,local_t,global_t>>(map, nx, ny, 4.0, -1.0, -1.0, -1.0, -1.0);
  // A_galeri->describe(*Teuchos::VerboseObjectBase::getDefaultOStream(), Teuchos::VERB_EXTREME);
  //--------------------------

  // reference solution and RHS
  auto x_ref = Teuchos::rcp(new Tpetra::Vector<scalar_t, local_t, global_t>(A->getDomainMap()));
  auto b = Teuchos::rcp(new Tpetra::Vector<scalar_t, local_t, global_t>(A->getRangeMap()));

  // Construct RHS from reference solution
  x_ref->randomize();
  A->apply(*x_ref, *b);

  // Storage for the solution
  auto x = Teuchos::rcp(new Tpetra::Vector<scalar_t, local_t, global_t>(A->getDomainMap()));

  auto lows_factory = Thyra::BelosLinearOpWithSolveFactory<scalar_t>();
  auto pl = Teuchos::rcp(new Teuchos::ParameterList());
  lows_factory.setParameterList(pl);
  lows_factory.setVerbLevel(Teuchos::VERB_HIGH);
  auto lows = lows_factory.createOp();

  auto rangespace = Thyra::tpetraVectorSpace<scalar_t>(A->getRangeMap());
  auto domainspace = Thyra::tpetraVectorSpace<scalar_t>(A->getDomainMap());
  const Teuchos::RCP<const Thyra::LinearOpBase<scalar_t>> A_thyra = Thyra::tpetraLinearOp(Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_t>>(rangespace), Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_t>>(domainspace), Teuchos::RCP<Tpetra::Operator<scalar_t,local_t,global_t>>(A));
  Thyra::initializeOp(lows_factory, A_thyra, lows.ptr());

  const Teuchos::RCP<Thyra::MultiVectorBase<scalar_t>> x_th = Thyra::tpetraVector<scalar_t,local_t,global_t,node_t>(domainspace, x);
  const Teuchos::RCP<Thyra::MultiVectorBase<scalar_t>> b_th = Thyra::tpetraVector<scalar_t,local_t,global_t,node_t>(rangespace, b);

  auto status = Thyra::solve(*lows, Thyra::NOTRANS, *b_th, x_th.ptr());
}

int main (int argc, char *argv[])
{
  assembly_time = Teuchos::TimeMonitor::getNewCounter("Assembly time");

  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  laplace2d(comm, 200, 20);
  laplace2d(comm, 200, 20);
  laplace2d(comm, 200, 20);

  Teuchos::TimeMonitor::report(std::cout);

  return 0;
}
