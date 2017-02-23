using Trilinos
using Base.Test
using MPI

lows_factory = Thyra.BelosLinearOpWithSolveFactory(Float64)
pl = Teuchos.ParameterList()
Thyra.setParameterList(lows_factory, pl)

Thyra.setVerbLevel(lows_factory, Teuchos.VERB_NONE)
lows = Thyra.createOp(lows_factory)

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const n = 20
rowmap = Tpetra.Map(n, 0, comm)
A = Tpetra.CrsMatrix(spdiagm((ones(n),), (0,)),rowmap)


rangespace = Thyra.tpetraVectorSpace(Tpetra.getRangeMap(A))
@show typeof(rangespace)
domainspace = Thyra.tpetraVectorSpace(Tpetra.getDomainMap(A))
A_thyra = Thyra.tpetraLinearOp(rangespace, domainspace, A)
Thyra.initializeOp(lows_factory, A_thyra, lows)

x = Tpetra.Vector(Tpetra.getDomainMap(A))
b = Tpetra.Vector(Tpetra.getRangeMap(A))

x_th = Thyra.tpetraVector(domainspace, x)
b_th = Thyra.tpetraVector(rangespace, b)

status = Thyra.solve(lows, Thyra.NOTRANS, b_th, x_th)