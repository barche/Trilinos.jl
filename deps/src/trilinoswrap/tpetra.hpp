#ifndef TRILINOS_JL_TPETRA_HPP
#define TRILINOS_JL_TPETRA_HPP

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>

#include "kokkos.hpp"

#include "jlcxx/jlcxx.hpp"

namespace jlcxx
{

template<> struct IsBits<Tpetra::ProfileType> : std::true_type {};

template<typename T1, typename T2, typename T3, typename T4> struct CopyConstructible<Tpetra::Vector<T1,T2,T3,T4>> : std::false_type {};

template<typename ST, typename LT, typename GT, typename NT> struct SuperType<Tpetra::CrsMatrix<ST,LT,GT,NT>> { typedef Tpetra::Operator<ST,LT,GT,NT> type; };
template<typename ST, typename LT, typename GT, typename NT> struct SuperType<Tpetra::Vector<ST,LT,GT,NT>> { typedef Tpetra::MultiVector<ST,LT,GT,NT> type; };

}

namespace trilinoswrap
{

typedef jlcxx::combine_types<jlcxx::ApplyType<Tpetra::RowMatrix>, scalars_t, local_ordinals_t, global_ordinals_t, kokkos_nodes_t> RowMatrixTypes;

jl_datatype_t* tpetra_operator_type();

}

#endif
