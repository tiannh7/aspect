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
implemented. Local-smoothing multigrid levels use the following preconditioner
approximations:

- use a positive representative `K * Delta t_m`, initially the geometric mean
  of the active minimum and maximum, in the pressure mass operators;
- retain the existing viscosity transfer and compressible deviatoric terms;
- evaluate `rho_ref * g * Delta t_m` and the radial unit vector at level
  quadrature points and include the mechanical radial velocity coupling in
  every level velocity block; and
- detect piecewise-constant reference-density contrasts on level faces and
  include `Delta rho * g * Delta t_m` in both the level velocity blocks and
  their Chebyshev diagonals; and
- when a free-surface boundary has an explicit stabilization density contrast,
  include its restoring face form in the level velocity block and Chebyshev
  diagonal using the same gravity/test and normal/trial ordering as the active
  operator.

The level-face construction represents interfaces that are faces of that
geometric level. It does not invent a sharp interface inside a coarse cell.
Consequently a strong material discontinuity should either be retained as a
level face or represented by an explicitly documented subcell projection.
These level terms change preconditioning quality, not the active linear
system. They were added after the refined-lateral PREM/VM5a discriminator made
the earlier omission resolution dependent.

Free-surface level terms currently require an explicit density contrast. A
boundary that obtains its contrast from the active material-model density is
left out of the level preconditioner until a general restriction of that
density to inactive geometric levels is implemented. The active operator is
unaffected by this limitation.

Custom meshes may be constructed entirely on level zero. In that case the
coarsest Chebyshev application is the complete velocity and Schur
preconditioner rather than the bottom of a multilevel V-cycle. For one-level
mechanical-mass-conservation models, the fixed coarse polynomial degree is 32
instead of 8. This preserves a linear preconditioner and the custom radial
grid; models with a genuine hierarchy and all other density-source laws retain
the established degree.

## Boundary-value and residual consistency

`correct_stokes_rhs()` manually applies the negative fine-grid operator to the
inhomogeneous constrained velocity. It includes the mechanical radial velocity
term and the internal-interface face term when present. The elastic
pressure mass and pressure-to-velocity radial terms do not contribute when the
constrained pressure is zero.

The matrix-free free-surface stabilization retains the assembled
nonsymmetric ordering. For coefficient $c$, gravity direction $\hat{g}$, and
boundary normal $\hat{n}$, the assembled form
$-c(\mathbf{w}\cdot\hat{g})(\mathbf{u}\cdot\hat{n})$ submits the value
$-c(\mathbf{u}\cdot\hat{n})\hat{g}$ in the active boundary operator. The
inhomogeneous constraint correction applies its negative. Transposing these
vectors is invisible on an undeformed sphere but inconsistent once
$\hat{g}$ and $\hat{n}$ are not parallel.

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
operator and local-smoothing velocity-block level operators apply this term.
The level diagonal includes the face contribution used by the Chebyshev
smoother.

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
correction, and internal-density-jump fine-grid and local-smoothing level face
operators, together with explicit-contrast free-surface level operators, are
implemented.
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
