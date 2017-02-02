using Trilinos
using Base.Test
using MPI

lows_factory = Thyra.BelosLinearOpWithSolveFactory()
pl = Teuchos.ParameterList()
Thyra.setParameterList(lows_factory, pl)

Thyra.setVerbLevel(lows_factory, Teuchos.VERB_EXTREME)
lows = Thyra.createOp(lows_factory)

comm = Teuchos.MpiComm(MPI.CComm(MPI.COMM_WORLD))
const n = 20
rowmap = Tpetra.Map(n, 0, comm)
A = Tpetra.CrsMatrix(spdiagm((2*ones(n-1),ones(n),3*ones(n-1)), (-1,0,1)),rowmap)

rangespace = Thyra.tpetraVectorSpace(Tpetra.getRangeMap(A))
domainspace = Thyra.tpetraVectorSpace(Tpetra.getDomainMap(A))
A_thyra = Thyra.tpetraLinearOp(rangespace, domainspace, A)
Thyra.initializeOp(lows_factory, A_thyra, lows)
