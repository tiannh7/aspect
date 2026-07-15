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

Local-smoothing `block GMG` is implemented. Its fine-grid operator contains the
elastic pressure mass, mechanical radial cell couplings, and
internal-density-jump face restoring term. Global-coarsening GMG remains
explicitly rejected until its separate coefficient-transfer path has been
implemented and tested.

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

Matrix-free assembly sets `rebuild_stokes_matrix = false`, so the assembled
right-hand side is retained while the pressure mass matrix is applied directly
by the fine-grid `StokesOperator`.

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

Active cell data therefore uses default-off selectors and quadrature data for
`K * Delta t_m`. Mechanical mass conservation additionally needs
`rho_ref * g * Delta t_m` and `e_r`. Values must come from
`DensitySourceManager` and the material model's `ElasticOutputs`, so timestep
zero uses the same initial elastic interval and the same reference-density
ownership as the assembled operator.

Elastic Stokes right-hand-side assembly explicitly requests viscosity. This is
required even when the matrix is not rebuilt because the viscoelastic
elastic-force update depends on the selected material averaging; requesting it
keeps assembled AMG and matrix-free GMG right-hand sides consistent.

## Block preconditioner

The positive pressure mass approximation used for the Schur complement is

```{math}
M_S \simeq \int_\Omega s^2
\left(\frac{1}{\eta}+\frac{1}{K\Delta t_m}\right)q p\,dV.
```

The existing $s^2/\eta$ term is unchanged for all other formulations. The
additional compliance term is enabled only for elastic pressure evolution.

For mechanical mass conservation, the active `BTBlockOperator` includes the
pressure-to-velocity radial coupling. The active `ABlockOperator` used by the
expensive preconditioner includes the nonsymmetric radial velocity coupling.
`stokes_A_block_is_symmetric()` consequently returns false for this
density-source law so that an inner nonsymmetric solver is selected.

## Local-smoothing levels

The fine-grid pressure, mechanical cell, and internal-face terms are
implemented. Local-smoothing multigrid levels deliberately remain simplified
preconditioner approximations:

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
inhomogeneous constrained velocity. It includes the mechanical radial velocity
term and the internal-interface face term when present. The elastic
pressure mass and pressure-to-velocity radial terms do not contribute when the
constrained pressure is zero.

The matrix-free solver's initial tolerance estimate uses the computed
pressure-row residual. Treating it as `norm(rhs_p)` would assume a zero pressure
block and is not valid once elastic pressure evolution adds a finite pressure
mass term. The finite pressure mass also removes the constant-pressure
nullspace, so the elastic-pressure pressure right-hand side is deliberately not
subjected to the compatibility projection used by singular pressure systems.

## Internal density jumps

Piecewise-constant PREM/VM5a reference-density tables derive internal density
jumps even when the explicit jump lists are empty. For each selected internal
face, the fine-grid operator applies

```{math}
+\int_\Gamma \Delta\rho\,g\Delta t_m
  (\mathbf w\cdot\mathbf e_r)(\mathbf v\cdot\mathbf e_r)\,dS.
```

The assembled path already supplies the committed-displacement and potential
right-hand-side terms on these faces. The matrix-free implementation uses the
interior-face loop and active-face coefficient tables. The fine-grid face
operator is implemented; multigrid levels still omit this term as a documented
preconditioner approximation.

## Compatibility and default behavior

No new user-facing experimental switch is needed. The consistency checks allow

```text
Mass conservation = elastic pressure evolution
Stokes solver type = block GMG
Stokes GMG type    = local smoothing
```

Global coarsening remains rejected with an explicit message. Existing defaults,
assembled AMG/direct behavior, all other mass-conservation formulations, and
elasticity-disabled models remain unchanged.

## Implementation status and scientific gates

The elastic pressure mass and Schur approximation, mechanical radial cell
couplings, nonsymmetric A-block selection, pressure residual, boundary-value
correction, and internal-density-jump fine-grid face operator are implemented.
Focused assembled-AMG/local-GMG regressions cover both a finite bulk-modulus
profile and a piecewise-constant internal density jump.

These implementation regressions are necessary but are not sufficient to call
PREM/VM5a production-ready. The remaining scientific acceptance gates are:

- same-grid AMG/GMG Love numbers and harmonic leakage agreeing within
  0.5 percent;
- the accepted uniform-compressible CitcomSVE trajectory remaining within
  0.5 percent through the short comparison interval;
- a short PREM/VM5a GMG--CitcomSVE comparison, including the target harmonic,
  non-target leakage, surface/CMB displacement, and stress, before any
  large-scale run is called ready.

The PREM/VM5a comparison must be run on G2 against the canonical CitcomSVE 3.0
executable. Until it passes, this backend is implemented and regression-tested
but not production-ready for those models.
