myname = splitdir(@__FILE__)[end]

excluded = []

for fname in readdir()
  if fname != myname && endswith(fname, ".jl") && fname ∉ excluded
    println("running test ", fname, "...")
    include(fname)
  end
end
