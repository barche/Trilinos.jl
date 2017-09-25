#ifndef TRILINOS_JL_IFPACK2_HPP
#define TRILINOS_JL_IFPACK2_HPP

#include "jlcxx/jlcxx.hpp"

#include <MueLu_TpetraOperator.hpp>

#include "tpetra.hpp"

namespace jlcxx
{

template<typename ST, typename LT, typename GT, typename NT> struct SuperType<MueLu::TpetraOperator<ST,LT,GT,NT>> { typedef Tpetra::Operator<ST,LT,GT,NT> type; };

}

#endif
