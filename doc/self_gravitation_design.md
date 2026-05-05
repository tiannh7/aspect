# Self-Gravitation with Topography Feedback — Design Document

## Physics

For GIA/surface-loading problems, the total surface mass anomaly modifies
the gravitational potential, which in turn modifies the effective surface load.

The total surface mass anomaly includes **both** the external load (e.g.,
mountain/ice) and the rock topography response (mesh deformation):

    σ_total = σ_load + Δρ_surf · h_rock
            = ρ_load · H_load + Δρ_surf · h_rock

The gravitational potential perturbation at degree l:

    δΦ_l = (4πG R) / (2l+1) · σ_total_l              (surface)
          + (4πG R) / (2l+1) · Δρ_cmb · h_cmb_l · (r_cmb/R)^(l+2)  (CMB)

The self-gravity ratio per degree:

    Rsg_l = 3 Δρ_surf / ((2l+1) ρ_mean)

Defining an effective height that combines load and deformation:

    h_effective = h_rock + σ_load / Δρ_surf

The self-gravity traction correction at the surface is:

    T_sg = Δρ_surf · g · Σ_l Rsg_l · h_effective_l · n̂

For a pure mountain load (ρ_load = Δρ_surf, no deformation yet):

    T_sg = Δρ_surf · g · Rsg · H_load · n̂   (outward → reduces effective load)
    T_eff = T_load · (1 - Rsg)               (load reduced by self-gravity)

At full isostatic equilibrium (h_rock ≈ -H_load):

    h_effective → 0, T_sg → 0   (no net mass anomaly, no self-gravity)

## Sign Convention

- `normal_vector` at outer surface: **radially outward**
- Inward load traction (T·n < 0) → positive surface mass → positive h_load
- Positive h_effective → outward self-gravity traction → reduces load ✓
- Negative h_effective → inward self-gravity traction → reduces restoring force ✓

## Architecture

### 1. SphericalHarmonicTransform / FourierTransform utility classes
Location: `include/aspect/utilities.h`

- 3D: SphericalHarmonicTransform for spherical shell geometry
- 2D: FourierTransform for annulus geometry
- Analysis:  f(θ,φ) → {f_lm^c, f_lm^s}
- Synthesis: {f_lm^c, f_lm^s} → f(θ,φ)

### 2. SelfGravitation boundary traction plugin
Location: `source/boundary_traction/self_gravitation.cc`

Registered as `"self gravitation"` boundary traction model.

In `update()` → `compute_self_gravity_correction()`:
  1. Loop over surface quadrature points
  2. Collect h_rock from mesh deformation (`height_above_reference_surface`)
  3. Get external load traction from boundary traction manager (subtract our
     own old contribution to isolate the load)
  4. Convert load to equivalent height: h_load = -T_load·n / (Δρ·g)
  5. SH/Fourier expand h_effective = h_rock + h_load
  6. Apply self-gravity kernel Rsg(l) per degree
  7. Add CMB contribution if enabled
  8. Store correction coefficients for `boundary_traction()` queries

In `boundary_traction()`:
  - Synthesize correction at the query point
  - Return: Δρ · g · correction · n̂

### 3. Self-gravity iteration
Integrated into the nonlinear Stokes iteration or time-stepping.

## Implementation Phases

Phase 1: SH/Fourier utility (analysis + synthesis for surface fields)
Phase 2: Self-gravitation boundary traction plugin
Phase 3: External load integration (total surface mass = load + topography)
Phase 4: Benchmarks (compare with CitcomSVE Heaviside=1 case)
