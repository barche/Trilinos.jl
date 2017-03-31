#ifndef TRILINOS_JL_TPETRA_HPP
#define TRILINOS_JL_TPETRA_HPP

#include <cxx_wrap.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>

namespace cxx_wrap
{

template<> struct IsBits<Tpetra::ProfileType> : std::true_type {};

template<typename T1, typename T2, typename T3, typename T4> struct CopyConstructible<Tpetra::Vector<T1,T2,T3,T4>> : std::false_type {};

template<typename ST, typename LT, typename GT, typename NT> struct SuperType<Tpetra::CrsMatrix<ST,LT,GT,NT>> { typedef Tpetra::Operator<ST,LT,GT,NT> type; };
template<typename ST, typename LT, typename GT, typename NT> struct SuperType<Tpetra::Vector<ST,LT,GT,NT>> { typedef Tpetra::MultiVector<ST,LT,GT,NT> type; };

}

#endif
