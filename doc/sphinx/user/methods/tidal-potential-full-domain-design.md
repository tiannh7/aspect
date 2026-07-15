# Compressible tidal-potential volume forcing design

## Scope and evidence

The Yuan et al. (2025) BM02 comparison uses a degree-2 tidal potential in a
compressible PREM/VM5a reference state.  The canonical CitcomSVE 3.0 weak form
adds the tidal potential through both density-interface tractions and the
volume term

```{math}
-\int_\Omega \rho_\mathrm{ref}\Phi_\mathrm{tidal}
\nabla\cdot\mathbf w\,dV.
```

ASPECT already assembles this term for its cached self-gravitational mass
potential, but the prescribed tidal potential is currently excluded from that
cache.  A 32-cubed GMG pilot consequently gives a reasonable radial response
but a much smaller horizontal response than the paired canonical CitcomSVE
run.  This design extends the existing full-domain path; it does not introduce
a second Stokes assembler or copy CitcomSVE architecture.

## Extension point

`PotentialFeedback::TidalPotential` will evaluate its prescribed external
solid harmonic at an arbitrary interior point,

```{math}
\frac{\Phi_\mathrm{tidal}(r,\theta,\phi)}{g_s}
= H_{lm}\left(\frac{r}{R}\right)^l Y_{lm}(\theta,\phi).
```

`SelfGravitation::full_domain_potential()` will add this value to the existing
self-gravitational mass potential.  The current Stokes cell and internal-sheet
assemblers will then use the combined potential without new assembly code.
Surface and CMB tractions remain on their existing path.

## Activation and compatibility

- No new parameter is required.
- The contribution is active only in 3-D spherical-shell models using the
  mechanical-mass-conservation density-source law and an enabled tidal
  potential.
- With tidal potential disabled, default behavior is unchanged.
- Incompressible models retain their boundary-only representation.
- This first patch does not alter rotational or reference-frame potentials.

## Validation

1. Add a focused unit test for the external solid-harmonic radial scaling and
   cosine/sine selection.
2. Run ASPECT indentation, the focused potential-feedback unit test, and the
   smallest relevant build target.
3. Rebuild on G2 and repeat only the 32-cubed, 12-rank, ten-step GMG BM02
   pilot against canonical CitcomSVE 3.0.
4. Accept the patch only if signed `h`, `k`, and `l` move toward CitcomSVE while
   harmonic leakage and potential-iteration convergence remain controlled.

The production R4 model remains an HPC-only follow-up after this short pair is
accepted.
