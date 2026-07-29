(sec:methods:glacial-isostatic-adjustment-design)=
# Glacial isostatic adjustment potential feedback design

This document defines the first coupled glacial-isostatic-adjustment (GIA)
implementation in ASPECT. The implementation targets the ICE-6G cases of
Zhong et al. (2022) and the ICE-6G_D cases of Yuan et al. (2025). It extends
the existing `potential feedback` boundary traction model; it does not add GIA
terms directly to the Stokes assembler and it does not reuse the passive
`sea level` postprocessor as a forcing model. The sea-level calculation follows
the single implementation in the canonical CitcomSVE 3.0 source tree.

The feature is disabled unless `glacial isostatic adjustment` is listed in
`Potential feedback/List of feedback mechanisms`. Existing self-gravity,
tidal, rotational-feedback, and non-GIA models therefore retain their current
behavior.

## Scope

The coupled implementation owns the following state and operations:

- a time-dependent gridded ice-thickness history with arbitrary stage times;
- a prescribed static or time-dependent ocean function;
- the canonical CitcomSVE sea-level equation;
- water-mass conservation through the spatially uniform barystatic constant;
- inward ice and ocean traction on the top boundary;
- inclusion of the same ice and ocean mass in self-gravity and rotational
  feedback;
- nonlinear convergence of the ocean load together with the existing
  self-gravity and rotational-potential iterations;
- checkpoint/restart of the current GIA load and iteration state.

The implementation requires a three-dimensional spherical-shell geometry, a
deforming top surface, and active self-gravity. Physical polar wander remains
the existing `rotational feedback` mechanism and is selected independently.
ASPECT nullspace removal remains a numerical reference-frame operation and is
not part of the GIA load calculation.

## Extension point and ownership

The new `PotentialFeedback::GlacialIsostaticAdjustment` class is a component
owned by `BoundaryTraction::PotentialFeedbackTraction`, alongside
`SelfGravitation` and `RotationalFeedback`.

This placement provides one owner for the coupled load state and gives the GIA
module access to the already selected top boundary and potential-iteration
settings. The unified boundary traction adapter returns the sum

```{math}
\mathbf t = \mathbf t_{\mathrm{ice+ocean}}
           + \mathbf t_{\mathrm{self\ gravity}}
           + \mathbf t_{\mathrm{rotation}}.
```

The GIA component is not registered as a second boundary traction plugin.
Instead, the adapter supplies its load traction to self-gravity and rotational
feedback through a narrow callback. This avoids recursive calls through the
boundary traction manager and ensures that both mechanisms use exactly the
same surface mass that is applied to Stokes.

The GIA component stores spherical-harmonic coefficients of surface mass per
unit area. Storing coefficients, rather than values tied to the current mesh,
allows the load to be evaluated consistently at Stokes quadrature points,
self-gravity sampling points, and after mesh refinement.

## Input histories

Ice and prescribed-ocean histories use structured latitude-longitude files and
a separate schedule. The explicit `citcomsve regular grid` data format reads
the canonical CitcomSVE `nlon nlat` header followed by longitude, latitude, and
field value in degrees. The loader converts to radian longitude-colatitude
coordinates, adds periodic longitude ghosts, and closes each pole with the
mean value of its nearest latitude row. The default `aspect structured data`
format remains available for standard ASPECT `# POINTS:` files. A schedule may
contain increasing elapsed model times and file numbers, or use the CitcomSVE
stage-age convention. Its header gives the number of stage intervals and is
followed by one more age row. CitcomSVE stage ages are converted to elapsed
time from the first stage; the second column, which records the number of
CitcomSVE time steps in a stage, does not control ASPECT's time step.

At a model time between two stages, the history loader linearly interpolates
the two gridded fields. It keeps the first field available as the reference
state. The loader clamps to the first or last field outside the schedule and
loads only the two bracketing grids needed at the current time.

## Load definition

Let `I` be ice thickness, `rho_i` ice density, `rho_w` water density,
`U` radial solid displacement, `N` geoid-height change, and `c` the
barystatic constant. Ice thickness is non-negative. The ice load is the change
in prescribed ice mass relative to either the first history file or zero
thickness, as selected by the input parameter.

With `Delta M_i` defined as current prescribed ice mass minus reference ice
mass, ice loss has `Delta M_i < 0` and contributes
`-Delta M_i/rho_w` to ocean water volume. Any grounding-line or shoreline
classification is part of the prescribed ice and ocean histories, matching the
canonical CitcomSVE workflow.

## Sea-level equation

The canonical CitcomSVE 3.0 form is

```{math}
L = O(N-U+c),
```

with

```{math}
c = \frac{-\Delta M_i/\rho_w
          -\int (N-U)O\,dS}
         {\int O\,dS}.
```

The ocean function may be static or prescribed at every ice stage. Input values
are clipped to the interval from zero to one, and stage interpolation may
produce fractional ocean coverage. This is algebraically equivalent to
CitcomSVE's separate static meltwater load and dynamic `N-U` ocean load, while
allowing the combined mass field to be passed consistently to ASPECT
self-gravity and rotational feedback.

The ocean mass anomaly is `rho_w L`. The total GIA surface mass anomaly is

```{math}
\sigma_{\mathrm{GIA}} = \Delta\sigma_i + \rho_w L,
```

and the applied top traction is

```{math}
\mathbf t_{\mathrm{GIA}} = -g\sigma_{\mathrm{GIA}}\mathbf n.
```

Positive surface mass therefore produces inward traction. The same traction is
converted back to mass by self-gravity and rotational feedback, so the applied
load, gravitational source, center-of-mass calculation, and inertia-tensor
calculation remain consistent.

## Coupled update order

At the beginning of a time step:

1. Load or interpolate the current ice and prescribed-ocean histories.
2. Form a first GIA load estimate using the committed displacement and the
   previous potential.
3. Update self-gravity from that load.
4. Update rotational feedback from the same load.
5. Recompute the sea-level equation using the new potential.

When potential iteration with Stokes is enabled, post-Stokes callbacks run in
this order:

1. self-gravity updates from the current GIA load and displacement predictor;
2. rotational feedback updates from the same mass distribution;
3. GIA updates `N`, `U`, `c`, and the ice-plus-ocean load for the prescribed
   current ocean function.

The solver continues while any active mechanism is unconverged. GIA
convergence is the relative L2 change of its spherical-harmonic surface-mass
coefficient vectors. The common potential-iteration tolerance, maximum
iteration count, and relaxation factor are used so that one parameter block
controls the coupled fixed-point iteration. Reaching the maximum iteration
count without meeting the GIA tolerance is an error, not successful
convergence.

For a global ocean, a single spherical harmonic of nonzero degree provides a
semi-analytic coupled test. The barystatic constant is degree zero and does not
enter this harmonic. If \(A_l\) is the relative sea-level response \(N-U\) per
unit surface-mass coefficient, then

```{math}
L_{lm} = A_l\left(\Delta\sigma_{i,lm}+\rho_w L_{lm}\right)
       = \frac{A_l}{1-\rho_w A_l}\Delta\sigma_{i,lm}.
```

This relation is the first oracle for checking the ASPECT fixed-point coupling
independently of shoreline coupling, moving coastlines, and realistic ice
histories. A nonuniform ocean function couples spherical harmonics and
therefore requires a matrix or pseudospectral numerical reference rather than
this scalar closed form.

## Geoid and displacement

`N` is synthesized from the current surface self-gravity potential height plus
the current rotational-potential height. It excludes committed local restoring
traction. `U` is the committed radial top-surface displacement plus the current
projected free-surface velocity increment during a post-Stokes update. This is
the same displacement predictor used by self-gravity and rotational feedback.

The center-of-mass convention is the existing
`Potential feedback/Reference frame/Degree 1 reference frame` selection. GIA
does not introduce a second degree-one correction.

## Checkpoint and restart

Boundary traction managers already serialize each active plugin through
`Plugins::InterfaceBase::save()` and `load()`. The unified potential-feedback
adapter uses this path to store the GIA barystatic constant, iteration counters,
reference values, and all current load coefficient vectors. Input grids are not
duplicated in the checkpoint; the history loader reconstructs its bracketing
files from model time, and the restored coefficients provide the exact load
state until the next coupled update.

## Diagnostics and acceptance

The existing `sea level` postprocessor will become a diagnostic view of the
active GIA provider. It will report or output at least relative sea level,
ocean function, ice load, ocean load, total GIA load, and the
barystatic constant. Each post-Stokes GIA/SLE update also reports prescribed
and applied ice mass, ocean-water mass, their absolute and relative closure
residuals, and the surface-load coefficient change. The relative water-mass
residual is normalized by
\(\int(|\Delta\sigma_i|+|\Delta\sigma_o|)\,dS\), rather than by either net
mass, so it remains meaningful for zero-mean spherical-harmonic loads.
When GIA is active, the `sea level` postprocessor reads both relative sea
level and geoid height directly from the unified potential-feedback provider;
the legacy `geoid` postprocessor is not a prerequisite.
The coupled update diagnostic additionally reports the degree-2, order-0
cosine coefficients of ice load, ocean load, total load, and relative sea
level. These four values make the global-ocean scalar oracle directly
testable without projecting a mesh-dependent visualization file.

Implementation acceptance requires, in this order:

1. a default-off regression proving unchanged existing behavior;
2. a small synthetic history-loader interpolation test;
3. a mass-conservation test for the canonical prescribed-ocean SLE;
4. a coupled low-resolution GIA smoke test with checkpoint/restart;
5. Zhong et al. (2022) and Yuan et al. (2025) production cases on HPC.

The production ICE-6G and ICE-6G_D calculations are not normal ASPECT tests and
are not to be run on macOS. macOS validation uses no more than 12 MPI ranks.
