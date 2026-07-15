(sec:methods:elastic-pressure-evolution-gmg)=
# Matrix-free GMG for elastic pressure evolution

This document defines the implementation boundary for using the matrix-free
`block GMG` Stokes solver with
`Mass conservation = elastic pressure evolution`. It uses the current ASPECT
interfaces
`Parameters<dim>::Formulation::MassConservation::elastic_pressure_evolution`
and
`Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation`.
It does not restore the retired `Allow experimental Citcom-style GMG` bypass
or any retired elastic-bulk formulation API.

The first supported backend is local-smoothing GMG. Global-coarsening GMG
remains explicitly rejected until its separate coefficient-transfer path has
been implemented and tested.

## Fine-grid operator

Let $s$ be ASPECT's pressure scaling, $K$ the elastic bulk modulus,
$\Delta t_m$ the effective mechanical time step, $\eta$ the viscosity,
$\rho_\mathrm{ref}$ the selected reference density, $g$ the gravity magnitude,
and $\mathbf e_r$ the radial unit vector. The active matrix-free operator must
reproduce the assembled operator term by term.

For every elastic-pressure-evolution model, the pressure row adds

```{math}
-\int_\Omega \frac{s^2}{K\Delta t_m}\,q p\,dV.
```

The existing assembled right-hand-side path already supplies

```{math}
-\int_\Omega \frac{s}{K\Delta t_m}\,q p^\mathrm{old}\,dV.
```

Matrix-free assembly sets `rebuild_stokes_matrix = false`, so the right-hand
side is retained while the pressure mass matrix is currently omitted. The
fine-grid `StokesOperator` must therefore apply the pressure mass term directly.

When `Density source law = mechanical mass conservation`, the fine-grid
operator additionally applies

```{math}
+\int_\Omega \frac{s\rho_\mathrm{ref}g}{K}
  (\mathbf w\cdot\mathbf e_r)p\,dV
-\int_\Omega \rho_\mathrm{ref}g\Delta t_m
  (\nabla\cdot\mathbf w)(\mathbf v\cdot\mathbf e_r)\,dV.
```

These are, respectively, an additional pressure-to-velocity coupling and a
nonsymmetric velocity-block coupling. The assembled path remains responsible
for the committed radial-displacement and full-domain-potential right-hand-side
terms. The matrix-free path must not assemble them a second time.

Active cell data therefore needs default-off selectors and quadrature data for
`K * Delta t_m`. Mechanical mass conservation additionally needs
`rho_ref * g * Delta t_m` and `e_r`. Values must come from
`DensitySourceManager` and the material model's `ElasticOutputs`, so timestep
zero uses the same initial elastic interval and the same reference-density
ownership as the assembled operator.

## Block preconditioner

The positive pressure mass approximation used for the Schur complement is

```{math}
M_S \simeq \int_\Omega s^2
\left(\frac{1}{\eta}+\frac{1}{K\Delta t_m}\right)q p\,dV.
```

The existing $s^2/\eta$ term is unchanged for all other formulations. The
additional compliance term is enabled only for elastic pressure evolution.

For mechanical mass conservation, the active `BTBlockOperator` must include
the pressure-to-velocity radial coupling. The active `ABlockOperator` used by
the expensive preconditioner must include the nonsymmetric radial velocity
coupling. `stokes_A_block_is_symmetric()` must consequently return false for
this density-source law so that an inner nonsymmetric solver is selected.

## Local-smoothing levels

The fine-grid operator is exact. Local-smoothing multigrid levels are allowed to
remain preconditioner approximations:

- use a positive representative `K * Delta t_m`, initially the geometric mean
  of the active minimum and maximum, in the pressure mass operators;
- retain the existing viscosity transfer and compressible deviatoric terms;
- omit the active radial volume coupling and internal-interface restoring terms
  from level operators in the first implementation.

Omitting these terms from multigrid levels changes preconditioning quality, not
the linear system. If convergence becomes resolution dependent, transferring
the coefficients to levels is a later, evidence-driven patch rather than part
of the first implementation.

## Boundary-value and residual consistency

`correct_stokes_rhs()` manually applies the negative fine-grid operator to the
inhomogeneous constrained velocity. It must include the mechanical radial
velocity term, and the internal-interface face term when present. The elastic
pressure mass and pressure-to-velocity radial terms do not contribute when the
constrained pressure is zero. An implementation may instead reject nonzero
prescribed velocity data until this correction is present, but it must not run
silently with an incomplete correction.

The matrix-free solver's initial tolerance estimate must also use the computed
pressure-row residual. Treating it as `norm(rhs_p)` assumes a zero pressure
block and is not valid once elastic pressure evolution adds a finite pressure
mass term.

## Internal density jumps

Piecewise-constant PREM/VM5a reference-density tables derive internal density
jumps even when the explicit jump lists are empty. For each selected internal
face, the fine-grid operator must apply

```{math}
+\int_\Gamma \Delta\rho\,g\Delta t_m
  (\mathbf w\cdot\mathbf e_r)(\mathbf v\cdot\mathbf e_r)\,dS.
```

The assembled path already supplies the committed-displacement and potential
right-hand-side terms on these faces. The matrix-free implementation should use
the interior-face loop and active-face coefficient tables; multigrid levels may
omit this term initially.

Until this fine-grid face operator exists, local-smoothing GMG must continue to
reject mechanical-mass-conservation models for which
`DensitySourceManager::has_internal_density_jumps()` is true. Removing the
general GMG guard without this narrower guard would silently omit PREM/VM5a
physics.

## Compatibility and default behavior

No new user-facing experimental switch is needed. After the corresponding
operators are implemented, the consistency checks may allow only

```text
Mass conservation = elastic pressure evolution
Stokes solver type = block GMG
Stokes GMG type    = local smoothing
```

Global coarsening remains rejected with an explicit message. Existing defaults,
assembled AMG/direct behavior, all other mass-conservation formulations, and
elasticity-disabled models remain unchanged.

## Implementation and test gates

Keep the implementation in small patches:

1. Add the elastic pressure mass term, Schur mass approximation, and a
   non-mechanical local-GMG regression.
2. Add the mechanical radial cell couplings, nonsymmetric A-block selection,
   pressure-row residual, and inhomogeneous-boundary correction.
3. Add the internal-density-jump fine-grid face operator and its focused test.
4. Enable the production PREM/VM5a local-GMG inputs only after the preceding
   tests pass.

Before removing each guard, require:

- unchanged default/no-op regression output;
- a small direct or assembled-AMG versus local-GMG comparison at timestep zero
  and at a nonzero timestep, including pressure and radial displacement;
- a nonzero mechanical radial-coupling test;
- a nonzero piecewise-constant internal-jump face test;
- an explicit failure test for global-coarsening GMG;
- same-grid AMG/GMG Love numbers and harmonic leakage agreeing within
  0.5 percent;
- the accepted uniform-compressible CitcomSVE trajectory remaining within
  0.5 percent through the short comparison interval;
- a short PREM/VM5a GMG--CitcomSVE comparison, including the target harmonic,
  non-target leakage, surface/CMB displacement, and stress, before any
  large-scale run is called ready.

Run `make indent`, compile the affected target, and execute only the focused
tests before the short scientific comparisons.
