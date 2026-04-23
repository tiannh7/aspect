# Self-Gravitation with Topography Feedback — Design Document

## Physics

For GIA/surface-loading problems, surface topography perturbations modify the
gravitational potential, which in turn modifies the effective surface load:

    δΦ_l = (4πG R ρ_surf) / (2l+1) · h_l      (surface contribution)
          + (4πG R ρ_cmb)  / (2l+1) · (r_cmb/R)^(l+2) · h_cmb_l  (CMB contribution)

The self-gravity ratio per degree:

    Rsg_l = 3 ρ_surf / ((2l+1) ρ_mean)    (surface)

The effective load at degree l becomes:

    F_eff_l = F_load_l · (1 - Rsg_l)   (first-order approximation)

For full self-gravity, an iterative solve is needed:
  1. Solve Stokes → velocity
  2. Update topography from velocity
  3. Recompute δΦ from new topography (via SH expansion)
  4. Update surface traction with δΦ contribution
  5. Repeat until convergence

## Architecture

### 1. SphericalHarmonicTransform utility class
Location: `include/aspect/utilities/spherical_harmonic_transform.h`

Efficient SH analysis and synthesis for surface fields:
- Analysis:  f(θ,φ) → {f_lm^c, f_lm^s}  for l=0..L_max
- Synthesis: {f_lm^c, f_lm^s} → f(θ,φ)

**Optimization over current geoid code:**
- Separate data collection (loop over cells once) from SH expansion
- Cache positions and field values
- For moderate L_max (<~64): direct Gauss-Legendre quadrature
- For high L_max: interpolate to regular grid + FFT

### 2. SelfGravitation assembler term
Location: `source/simulator/assemblers/stokes.cc` (additional term)

Adds the self-gravity correction as a surface traction:

    δT_n = ρ_surf · δΦ / R

on the top/bottom boundaries. This modifies the Stokes RHS.

### 3. Self-gravity iteration
Triggered by `set_assemblers` signal or integrated into the nonlinear iteration.

## Implementation Phases

Phase 1: SH utility (analysis + synthesis for surface fields)
Phase 2: Self-gravitation assembler (boundary traction correction)
Phase 3: Iteration loop integration
Phase 4: Benchmarks (compare with CitcomSVE Heaviside=1 case)
