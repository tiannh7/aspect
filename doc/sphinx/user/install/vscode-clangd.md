# VS Code and clangd

ASPECT ships a clangd configuration for C++ language servers such as the VS Code
clangd extension. Configure ASPECT with `ASPECT_SETUP_CLANGD=ON` to make the
current build directory produce a clangd-friendly compilation database.

Create or reconfigure your normal build directory with the compiler and deal.II
installation used on your machine. For example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DASPECT_SETUP_CLANGD=ON \
  -Ddeal.II_DIR=/path/to/deal.II/lib/cmake/deal.II
```

If ASPECT is built with MPI, choose the MPI C++ wrapper as the C++ compiler,
for example by configuring CMake with `CXX=mpicxx` or
`-DCMAKE_CXX_COMPILER=/path/to/mpicxx`.

The `ASPECT_SETUP_CLANGD` option enables `CMAKE_EXPORT_COMPILE_COMMANDS`,
disables precompiled headers and unity builds, and creates
`compile_commands.json` in the source directory as a symlink to the current
build directory's compilation database. clangd can then discover the database
automatically, independent of whether the build directory is named `build`,
`build-release`, or something else.

The VS Code workspace settings allow clangd to query common C++ and MPI compiler
wrappers for system include paths. Do not commit machine-specific compiler or
deal.II paths to the workspace settings; keep them in your shell environment,
CMake kit, or local `CMakeUserPresets.json`.
