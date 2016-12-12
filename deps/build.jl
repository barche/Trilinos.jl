using BinDeps
using CxxWrap

TRILINOS_ROOT = get(ENV, "TRILINOS_ROOT", "")

@static if is_windows()
  build_on_windows = false
  # prefer building if requested
  if TRILINOS_ROOT != ""
    build_on_windows = true
    saved_defaults = deepcopy(BinDeps.defaults)
    empty!(BinDeps.defaults)
    append!(BinDeps.defaults, [BuildProcess])
  end
end

@BinDeps.setup

cxx_wrap_dir = Pkg.dir("CxxWrap","deps","usr","lib","cmake")

trilinoswrap = library_dependency("trilinoswrap", aliases=["libtrilinoswrap"])

cmake_prefix = TRILINOS_ROOT

prefix=joinpath(BinDeps.depsdir(trilinoswrap),"usr")
trilinoswrap_srcdir = joinpath(BinDeps.depsdir(trilinoswrap),"src","trilinoswrap")
trilinoswrap_builddir = joinpath(BinDeps.depsdir(trilinoswrap),"builds","trilinoswrap")
lib_prefix = @static is_windows() ? "" : "lib"
lib_suffix = @static is_windows() ? "dll" : (@static is_apple() ? "dylib" : "so")

makeopts = ["--", "-j", "$(Sys.CPU_CORES+2)"]

# Set generator if on windows
genopt = "Unix Makefiles"
@static if is_windows()
  makeopts = "--"
  if Sys.WORD_SIZE == 64
    genopt = "Visual Studio 14 2015 Win64"
    cmake_prefix = joinpath(TRILINOS_ROOT, "msvc2015_64")
  else
    genopt = "Visual Studio 14 2015"
    cmake_prefix = joinpath(TRILINOS_ROOT, "msvc2015")
  end
end

qml_steps = @build_steps begin
	`cmake -G "$genopt" -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_PREFIX_PATH="$cmake_prefix" -DCxxWrap_DIR="$cxx_wrap_dir" $trilinoswrap_srcdir`
	`cmake --build . --config Release --target install $makeopts`
end

# If built, always run cmake, in case the code changed
if isdir(trilinoswrap_builddir)
  BinDeps.run(@build_steps begin
    ChangeDirectory(trilinoswrap_builddir)
    qml_steps
  end)
end

provides(BuildProcess,
  (@build_steps begin
    CreateDirectory(trilinoswrap_builddir)
    @build_steps begin
      ChangeDirectory(trilinoswrap_builddir)
      FileRule(joinpath(prefix,"lib", "$(lib_prefix)trilinoswrap.$lib_suffix"),qml_steps)
    end
  end),trilinoswrap)

#deps = [trilinoswrap]
#provides(Binaries, Dict(URI("https://github.com/barche/QML.jl/releases/download/v0.2.0/QML-julia-$(VERSION.major).$(VERSION.minor)-win$(Sys.WORD_SIZE).zip") => deps), os = :Windows)

@BinDeps.install Dict([(:trilinoswrap, :_l_qml_wrap)])

@static if is_windows()
  if build_on_windows
    empty!(BinDeps.defaults)
    append!(BinDeps.defaults, saved_defaults)
  end
end
