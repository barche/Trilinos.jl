using MPI

myname = splitdir(@__FILE__)[end]

intesting = true

MPI.Init()

excluded = []

for fname in readdir()
  if fname != myname && endswith(fname, ".jl") && fname ∉ excluded
    println("running test ", fname, "...")
    include(fname)
  end
end

MPI.Finalize()
