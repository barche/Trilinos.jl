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


Teuchos::RCP<Teuchos::Time> graph_time;
Teuchos::RCP<Teuchos::Time> fill_time;
Teuchos::RCP<Teuchos::Time> source_time;
Teuchos::RCP<Teuchos::Time> dirichlet_time;
Teuchos::RCP<Teuchos::Time> check_time;

//#include <Galeri_XpetraMatrixTypes.hpp>

typedef double scalar_t;
typedef int local_t;
typedef int64_t global_t;

struct GlobalIndexing
{
  typedef global_t IdxT;

  template<typename MT, typename GT, typename IT, typename VT>
  inline static local_t replace_values(const MT& A, const GT gid, const IT& indices, const VT& values)
  {
    return A.replaceGlobalValues(gid, indices, values);
  }

  template<typename MapT, typename LT>
  inline static global_t global_element(const MapT& m, const LT i)
  {
    return m.getGlobalElement(i);
  }

  template<typename MapT, typename LT>
  inline static global_t local_element(const MapT& m, const LT i)
  {
    return m.getLocalElement(i);
  }

  template<typename MapT, typename LT>
  inline static global_t is_node_global_element(const MapT& m, const LT i)
  {
    return m.isNodeGlobalElement(i);
  }
};

struct LocalIndexing
{
  typedef local_t IdxT;

  template<typename MT, typename GT, typename IT, typename VT>
  inline static local_t replace_values(const MT& A, const GT gid, const IT& indices, const VT& values)
  {
    return A.replaceLocalValues(gid, indices, values);
  }

  template<typename MapT, typename LT>
  inline static global_t global_element(const MapT& m, const LT i)
  {
    return i;
  }

  template<typename MapT, typename LT>
  inline static global_t local_element(const MapT& m, const LT i)
  {
    return i;
  }

  template<typename MapT, typename LT>
  inline static global_t is_node_global_element(const MapT& m, const LT i)
  {
    return true;
  }
};

struct CartesianGrid
{
  global_t nx;
  global_t ny;
  scalar_t h;

  global_t nnodes() const { return nx*ny; }

  std::pair<global_t,global_t> xyindices(const global_t i) const
  {
    const global_t ix = i % nx;
    return std::make_pair(ix, (i-ix) / nx);
  }

  std::pair<scalar_t,scalar_t> coordinates(const global_t i) const
  {
    const auto p = xyindices(i);
    const global_t ix = p.first;
    const global_t iy = p.second;
    const scalar_t x_center = h*(nx-1)/2;
    const scalar_t y_center = h*(ny-1)/2;
    return std::make_pair(ix*h - x_center, iy*h - y_center);
  }
};

template<typename IndexingT>
struct Laplace2D
{

  template<typename IdxT>
  static local_t laplace2d_indices(IdxT inds_array[5], const global_t i, const CartesianGrid& g)
  {
    const auto p = g.xyindices(i);
    const IdxT ix = p.first;
    const IdxT iy = p.second;

    local_t n_inds = 0;
    inds_array[n_inds] = i;
    if(iy != g.ny-1)
    {
        n_inds += 1;
        inds_array[n_inds] = i+g.nx;
    }
    if(ix != 0)
    {
        n_inds += 1;
        inds_array[n_inds] = i-1;
    }
    if(iy != 0)
    {
        n_inds += 1;
        inds_array[n_inds] = i-g.nx;
    }
    if(ix != g.nx-1)
    {
        n_inds += 1;
        inds_array[n_inds] = i+1;
    }
    return n_inds+1;
  }

  template<typename VectorT>
  static void set_source_term(VectorT& b, const CartesianGrid& g)
  {
    Teuchos::TimeMonitor local_timer(*source_time);
    const auto& rowmap = *b.getMap();
    auto b_view = b.template getLocalView<typename VectorT::dual_view_type::t_dev>();
    const local_t n_my_elms = rowmap.getNodeNumElements();
    for(local_t i = 0; i != n_my_elms; ++i)
    {
      const global_t gid = IndexingT::global_element(rowmap,i);
      const auto p = g.coordinates(gid);
      const scalar_t x = p.first;
      const scalar_t y = p.second;
      b_view(i,0) = 2*g.h*g.h*((1-x*x)+(1-y*y));
    }
  }

  template<typename GraphT>
  static void graph_laplace2d(GraphT& A, const CartesianGrid& g)
  {
    Teuchos::TimeMonitor local_timer(*graph_time);
    const auto& rowmap = *A.getRowMap();
    local_t n_my_elms = rowmap.getNodeNumElements();

    // storage for the per-row values
    global_t row_indices[5] = {0,0,0,0,0};

    for(local_t i = 0; i != n_my_elms; ++i)
    {
      const global_t global_row = IndexingT::global_element(rowmap,i);
      const local_t row_n_elems = laplace2d_indices(row_indices, global_row, g);
      A.insertGlobalIndices(global_row, Teuchos::ArrayView<global_t>(row_indices,row_n_elems));
    }
  }

  template<typename MatrixT>
  static void fill_laplace2d(MatrixT& A, const CartesianGrid& g)
  {
    Teuchos::TimeMonitor local_timer(*fill_time);
    const auto& rowmap = *A.getRowMap();
    const local_t n_my_elms = rowmap.getNodeNumElements();

    // storage for the per-row values
    typename IndexingT::IdxT row_indices[5] = {0,0,0,0,0};
    scalar_t row_values[5] = {4.0,-1.0,-1.0,-1.0,-1.0};

    for(local_t i = 0; i != n_my_elms; ++i)
    {
      const global_t global_row = IndexingT::global_element(rowmap,i);
      const local_t row_n_elems = laplace2d_indices(row_indices, global_row, g);
      row_values[0] = 4.0 - (5-row_n_elems);
      IndexingT::replace_values(A, global_row, Teuchos::ArrayView<typename IndexingT::IdxT>(row_indices,row_n_elems), Teuchos::ArrayView<scalar_t>(row_values,row_n_elems));
    }
  }

  template<typename MatrixT, typename VectorT>
  static void set_dirichlet(MatrixT& A, VectorT& b, const CartesianGrid& g)
  {
    Teuchos::TimeMonitor local_timer(*dirichlet_time);
    const auto& rowmap = *A.getRowMap();
    typename IndexingT::IdxT row_indices[5] = {0,0,0,0,0};
    scalar_t row_values[5] = {1.0,0.0,0.0,0.0,0.0};

    auto b_view = b.template getLocalView<typename VectorT::dual_view_type::t_dev>();

    // reserve space for the boundary nodes
    std::vector<global_t> boundary_gids;
    boundary_gids.reserve((2*(g.nx+g.ny)));

    // left and right
    for(global_t iy = 0; iy != g.ny; ++iy)
    {
      boundary_gids.push_back(iy*g.nx);
      boundary_gids.push_back(boundary_gids.back() + g.nx-1);
    }

    // top and bottom
    for(global_t ix = 0; ix != g.nx; ++ix)
    {
      boundary_gids.push_back(ix);
      boundary_gids.push_back(ix + (g.ny-1)*g.nx);
    }

    // Apply BC
    for(const global_t gid : boundary_gids)
    {
      const auto p = g.coordinates(gid);
      const scalar_t x = p.first;
      const scalar_t y = p.second;
      if(IndexingT::is_node_global_element(rowmap,gid))
      {
        const local_t row_n_elems = laplace2d_indices(row_indices, gid, g);
        IndexingT::replace_values(A, gid, Teuchos::ArrayView<typename IndexingT::IdxT>(row_indices,row_n_elems), Teuchos::ArrayView<scalar_t>(row_values,row_n_elems));
        b_view(IndexingT::local_element(rowmap,gid),0) = (1-x*x)*(1-y*y);
      }
    }
  }

  template<typename VectorT>
  static bool check_solution(VectorT& sol, const CartesianGrid& g)
  {
    Teuchos::TimeMonitor local_timer(*check_time);
    const auto& solmap = *sol.getMap();
    const local_t n_my_elms = solmap.getNodeNumElements();

    auto solview = sol.template getLocalView<typename VectorT::dual_view_type::t_dev>();
    global_t result = 0;
    for(local_t i = 0; i != n_my_elms; ++i)
    {
      const global_t gid = IndexingT::global_element(solmap,i);
      const auto p = g.coordinates(gid);
      const scalar_t x = p.first;
      const scalar_t y = p.second;
      result += std::abs(solview(i,0) - (1-x*x)*(1-y*y)) > 1e-7;
    }
    return result == 0;
  }

  template<typename comm_t>
  static void laplace2d(comm_t comm, const CartesianGrid& g, const bool solve = false)
  {
    auto map = Teuchos::rcp(new Tpetra::Map<local_t,global_t>(g.nnodes(), 0, comm));
    typedef Tpetra::Map<local_t,global_t>::node_type node_t;

    auto matrix_graph = Teuchos::rcp(new Tpetra::CrsGraph<local_t,global_t>(map, 0));
    graph_laplace2d(*matrix_graph, g);
    matrix_graph->fillComplete();

    auto b = Teuchos::rcp(new Tpetra::Vector<scalar_t, local_t, global_t>(matrix_graph->getRangeMap()));
    set_source_term(*b,g);

    // Matrix construction and fill
    auto A = Teuchos::rcp(new Tpetra::CrsMatrix<scalar_t,local_t,global_t>(matrix_graph));
    A->resumeFill();
    fill_laplace2d(*A, g);
    set_dirichlet(*A,*b,g);
    A->fillComplete();

    if(solve)
    {
      //b->describe(*Teuchos::VerboseObjectBase::getDefaultOStream(), Teuchos::VERB_EXTREME);
      //A->describe(*Teuchos::VerboseObjectBase::getDefaultOStream(), Teuchos::VERB_EXTREME);
      // Storage for the solution
      auto x = Teuchos::rcp(new Tpetra::Vector<scalar_t, local_t, global_t>(A->getDomainMap()));

      auto lows_factory = Thyra::BelosLinearOpWithSolveFactory<scalar_t>();
      auto pl = Teuchos::rcp(new Teuchos::ParameterList());
      pl->set("Solver Type", "Block GMRES");
      Teuchos::ParameterList& solver_pl = pl->sublist("Solver Types").sublist("Block GMRES");
      solver_pl.set("Convergence Tolerance", 1e-12);
      solver_pl.set("Maximum Iterations", 1000);
      solver_pl.set("Num Blocks", 1000);
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

      if(!check_solution(*x,g))
      {
        std::cout << "ERROR: wrong solution!" << std::endl;
      }
    }
  }

  static void do_run()
  {
    graph_time = Teuchos::TimeMonitor::getNewCounter("Graph construction time");
    fill_time = Teuchos::TimeMonitor::getNewCounter("Matrix fill time");
    source_time = Teuchos::TimeMonitor::getNewCounter("Source term time");
    dirichlet_time = Teuchos::TimeMonitor::getNewCounter("Dirichlet time");
    check_time = Teuchos::TimeMonitor::getNewCounter("Check solution time");

    Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

    if(comm->getSize() != 1 && typeid(IndexingT) == typeid(LocalIndexing))
    {
      throw std::runtime_error("Local indexing is not supported in parallel runs");
    }

    CartesianGrid grid = {101,41,1.0/50.0};

    laplace2d(comm, grid);
    Teuchos::TimeMonitor::report(std::cout);
    Teuchos::TimeMonitor::zeroOutTimers();
    laplace2d(comm, grid);
    Teuchos::TimeMonitor::report(std::cout);
    Teuchos::TimeMonitor::zeroOutTimers();
    laplace2d(comm, grid, true);
    Teuchos::TimeMonitor::report(std::cout);

  }

};

int main (int argc, char *argv[])
{
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);
  if(argc == 2 && std::string(argv[1]) == "local")
  {
    std::cout << "local indexing" << std::endl;
    Laplace2D<LocalIndexing>::do_run();
  }
  else
  {
    Laplace2D<GlobalIndexing>::do_run();
  }

  return 0;
}
