#ifndef TRILINOS_JL_IFPACK2_HPP
#define TRILINOS_JL_IFPACK2_HPP

#include <Ifpack2_Factory.hpp>

#include "tpetra.hpp"

#include "jlcxx/jlcxx.hpp"

namespace jlcxx
{

template<typename ST, typename LT, typename GT, typename NT> struct SuperType<Ifpack2::Preconditioner<ST,LT,GT,NT>> { typedef Tpetra::Operator<ST,LT,GT,NT> type; };

}

#endif
