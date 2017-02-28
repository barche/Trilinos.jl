module Thyra
using CxxWrap, MPI, Compat
import .._l_trilinos_wrap
import ..RCPWrappable
import ..RCPAssociative
import ..Teuchos
import ..Tpetra

export LinearOpWithSolve

registry = load_modules(_l_trilinos_wrap)

wrap_module_types(registry)

CxxWrap.argument_overloads{ScalarT}(t::Type{Teuchos.RCP{LinearOpBase{ScalarT}}}) = [Teuchos.RCP]
CxxWrap.argument_overloads{ScalarT}(t::Type{Teuchos.RCP{VectorSpaceBase{ScalarT}}}) = [Teuchos.RCP]
CxxWrap.argument_overloads{ScalarT}(t::Type{Teuchos.RCP{MultiVectorBase{ScalarT}}}) = [Teuchos.RCP]
CxxWrap.argument_overloads{ScalarT}(t::Type{Teuchos.RCPPtr{MultiVectorBase{ScalarT}}}) = [Teuchos.RCP]
CxxWrap.argument_overloads{ScalarT}(t::Type{MultiVectorBase{ScalarT}}) = [Teuchos.RCP]

wrap_module_functions(registry)

# convenience Union, types are scalar, local ordinal, global ordinal and node.
typealias TpetraMatrixUnion{ST,LT,GT,NT} Union{Teuchos.RCP{Tpetra.CrsMatrix{ST,LT,GT,NT}}, Tpetra.CrsMatrix{ST,LT,GT,NT}}
typealias TpetraVectorUnion{ST,LT,GT,NT} Union{Teuchos.RCP{Tpetra.Vector{ST,LT,GT,NT}}, Tpetra.Vector{ST,LT,GT,NT}}
typealias ParameterListUnion Union{Teuchos.RCP{Teuchos.ParameterList}, Teuchos.ParameterList}

@compat immutable LinearOpWithSolve{ST,LT,GT,NT}
  lows::Teuchos.RCP{Thyra.LinearOpWithSolveBase{ST}}
  domainmap::Teuchos.RCP{Tpetra.Map{LT,GT,NT}}
  rangespace::Teuchos.RCP{Thyra.TpetraVectorSpace{ST,LT,GT,NT}}
  domainspace::Teuchos.RCP{Thyra.TpetraVectorSpace{ST,LT,GT,NT}}

  function (::Type{LinearOpWithSolve{ST,LT,GT,NT}}){ST,LT,GT,NT}(A::TpetraMatrixUnion{ST,LT,GT,NT}, parameters::ParameterListUnion, verbosity::Teuchos.EVerbosityLevel)
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

function LinearOpWithSolve{ST,LT,GT,NT}(A::TpetraMatrixUnion{ST,LT,GT,NT}, parameters::ParameterListUnion=Teuchos.ParameterList(), verbosity::Teuchos.EVerbosityLevel=Teuchos.VERB_NONE)
  return LinearOpWithSolve{ST,LT,GT,NT}(A,parameters,verbosity)
end

import Base: \

function \{ST,LT,GT,NT}(A::LinearOpWithSolve{ST,LT,GT,NT}, b::TpetraVectorUnion{ST,LT,GT,NT})
  x = Tpetra.Vector(A.domainmap)
  x_th = Thyra.tpetraVector(A.domainspace, x)
  b_th = Thyra.tpetraVector(A.rangespace, b)
  status = Thyra.solve(A.lows, Thyra.NOTRANS, b_th, x_th)
  return x
end

end # module
