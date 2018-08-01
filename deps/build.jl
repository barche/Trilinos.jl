using BinaryProvider

const JLTRILINOS_DIR = get(ENV, "JLTRILINOS_DIR", "")
const prefix = Prefix(JLTRILINOS_DIR == "" ? !isempty(ARGS) ? ARGS[1] : joinpath(@__DIR__,"usr") : JLTRILINOS_DIR)

products = Product[
    LibraryProduct(prefix, "libjltrilinos", :libjltrilinos)
]

if any(!satisfied(p; verbose=true) for p in products)
    # TODO: Add download code here
end

write_deps_file(joinpath(@__DIR__, "deps.jl"), products)
