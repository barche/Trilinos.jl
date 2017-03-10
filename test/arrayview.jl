using Trilinos
using Base.Test
using MPI
import Trilinos.Testing.arrayview_sum

a = collect(1.0:10.0)
s = sum(a)
s_half = sum(a[1:5])

@test arrayview_sum(Teuchos.ArrayView(a)) == s
@test arrayview_sum(a) == s
@test arrayview_sum(a[1:5]) == s_half
@test arrayview_sum(@view a[1:5]) == s_half
@test arrayview_sum(Teuchos.ArrayView(a,5)) == s_half
