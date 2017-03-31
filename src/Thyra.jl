module Thyra
using CxxWrap, MPI, Compat
import .._l_trilinos_wrap
import ..CxxUnion
import ..Teuchos
import ..Tpetra

export LinearOpWithSolve

registry = load_modules(_l_trilinos_wrap)

wrap_module_types(registry)

wrap_module_functions(registry)

@compat immutable LinearOpWithSolve{ST,LT,GT,NT}
  lows::CxxWrap.SmartPointer{Thyra.LinearOpWithSolveBase{ST}}
  domainmap::CxxWrap.SmartPointer{Tpetra.Map{LT,GT,NT}}
  rangespace::CxxWrap.SmartPointer{Thyra.TpetraVectorSpace{ST,LT,GT,NT}}
  domainspace::CxxWrap.SmartPointer{Thyra.TpetraVectorSpace{ST,LT,GT,NT}}

  function (::Type{LinearOpWithSolve{ST,LT,GT,NT}}){ST,LT,GT,NT}(A::CxxUnion{Tpetra.CrsMatrix{ST,LT,GT,NT}}, parameters::CxxUnion{Teuchos.ParameterList}, verbosity::Teuchos.EVerbosityLevel)
    lows_factory = Thyra.BelosLinearOpWithSolveFactory(ST)
    Thyra.setParameterList(lows_factory, parameters)
    Thyra.setVerbLevel(lows_factory, verbosity)

    lows = Thyra.createOp(lows_factory)
    domainmap = Tpetra.getDomainMap(A)
    rangespace = Thyra.tpetraVectorSpace(Tpetra.getRangeMap(A))
    domainspace = Thyra.tpetraVectorSpace(domainmap)
    A_thyra = Thyra.tpetraLinearOp(rangespace, domainspace, A)
    Thyra.initializeOp(lows_factory, A_thyra, lows)

    return new{ST,LT,GT,NT}(lows, domainmap, rangespace, domainspace)
  end
end

function LinearOpWithSolve{ST,LT,GT,NT}(A::CxxUnion{Tpetra.CrsMatrix{ST,LT,GT,NT}}, parameters::CxxUnion{Teuchos.ParameterList}=Teuchos.ParameterList(), verbosity::Teuchos.EVerbosityLevel=Teuchos.VERB_NONE)
  return LinearOpWithSolve{ST,LT,GT,NT}(A,parameters,verbosity)
end

import Base: \

function \{ST,LT,GT,NT}(A::LinearOpWithSolve{ST,LT,GT,NT}, b::CxxUnion{Tpetra.Vector{ST,LT,GT,NT}})
  x = Tpetra.Vector(A.domainmap)
  x_th = Thyra.tpetraVector(A.domainspace, x)
  b_th = Thyra.tpetraVector(A.rangespace, b)
  status = Thyra.solve(A.lows, Thyra.NOTRANS, b_th, x_th)
  return x
end

end # module
