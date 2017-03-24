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
examples_srcdir = joinpath(BinDeps.depsdir(trilinoswrap),"src","upstream_examples")
examples_builddir = joinpath(BinDeps.depsdir(trilinoswrap),"builds","upstream_examples")
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

build_type = get(ENV, "CXXWRAP_BUILD_TYPE", "Release")

trilinos_steps = @build_steps begin
	`cmake -G "$genopt" -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_PREFIX_PATH="$cmake_prefix" -DCxxWrap_DIR="$cxx_wrap_dir" -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc $trilinoswrap_srcdir`
	`cmake --build . --config $build_type --target install $makeopts`
end

examples_steps = @build_steps begin
	`cmake -G "$genopt" -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_PREFIX_PATH="$cmake_prefix" -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc $examples_srcdir`
	`cmake --build . --config $build_type $makeopts`
end

# If built, always run cmake, in case the code changed
if isdir(trilinoswrap_builddir)
  BinDeps.run(@build_steps begin
    ChangeDirectory(trilinoswrap_builddir)
    trilinos_steps
  end)
end

if(!isdir(examples_builddir))
  BinDeps.run(@build_steps begin
    CreateDirectory(examples_builddir)
    @build_steps begin
      ChangeDirectory(examples_builddir)
      examples_steps
    end
  end)
else
  BinDeps.run(@build_steps begin
    ChangeDirectory(examples_builddir)
    examples_steps
  end)
end

provides(BuildProcess,
  (@build_steps begin
    CreateDirectory(trilinoswrap_builddir)
    @build_steps begin
      ChangeDirectory(trilinoswrap_builddir)
      FileRule(joinpath(prefix,"lib", "$(lib_prefix)trilinoswrap.$lib_suffix"),trilinos_steps)
    end
  end),trilinoswrap)

@BinDeps.install Dict([(:trilinoswrap, :_l_trilinos_wrap)])

@static if is_windows()
  if build_on_windows
    empty!(BinDeps.defaults)
    append!(BinDeps.defaults, saved_defaults)
  end
end
