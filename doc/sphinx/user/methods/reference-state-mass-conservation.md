(sec:methods:reference-state-mass-conservation-design)=
# Reference-state mass-conservation design

This document is the implementation design and repository audit for Phase II
of the unified density-source framework. It was written before changing C++
code. The implementation branch starts at `dac7e4fbe3` on
`feature/unified-density-source-framework`; that branch contains both Phase-I
commits `8d5065d450` and `dac7e4fbe3`.

The scope is a smooth reference state and linearized mechanical density
perturbations. PREM, tabulated profiles, sharp internal interfaces, total-field
gravity, GIA, sea level, tides, and polar-wander changes are excluded.

## 1. Phase-I density-source architecture

Material models own physical material-property evaluation. In particular,
`MaterialModelOutputs::densities` means physical material density and is not a
perturbation. `DensitySourceManager<dim>` is simulator-owned and exposed
through `SimulatorAccess`; it owns reference-density selection and typed views
for the volume sources consumed by Stokes and perturbation self-gravity.

The independent Phase-I choices are:

- reference models: `none`, `constant`, and
  `frozen initial lateral average`;
- source laws: `legacy`, `material density`, `material minus reference`, and
  `zero volume perturbation`.

The default `none + legacy` preserves historical Stokes and self-gravity
semantics. Surface and CMB sheets, their density-jump restoring terms, and FSSA
remain outside the volume-source manager.

## 2. Compressible-Maxwell branch audit

The local compressible work exists on three branch families. Commits
`b336cf5861` and `2ecfdafce6` have identical patches on different bases;
`ae4f615c3b` and `7c301f6be3` are likewise patch-identical. The
`codex/compressible-on-fix-zhong2022` family is the only one based directly on
the validated Zhong2022 commit `a4ea2d441d`.

| Commit | Branch | Principal files | Physical purpose and dependencies | Density-manager conflict | Decision |
| --- | --- | --- | --- | --- | --- |
| `b336cf5861` | `codex/compressible-maxwell-from-zhong` ancestry | viscoelastic material/rheology, Stokes assemblers, matrix-free solvers, profile utilities, tests | Umbrella prototype: compressible EOS, bulk and Lamé inputs, scalar `ve_pressure`, ascii profiles, Citcom bulk/restoring operators, GMG | Predates Phase I; contains a material-private constant Stokes reference subtraction and duplicates source ownership | Do not port commit. Reimplement only uniform bulk-modulus output and a focused baseline test |
| `2ecfdafce6` | `codex/compressible-on-fix-zhong2022` | same patch as `b336cf5861` | Same umbrella prototype on the correct Zhong2022 base | Same conflict; also combines generic and benchmark-specific changes | Selective source reference only; no cherry-pick |
| `5b6ce4df16` | `codex/backup-compressible-20260710` | parameters, assembly checks, design notes | Feasibility guards layered on `b336cf5861` | Depends on the rejected umbrella prototype | Do not port separately |
| `ae4f615c3b` | `codex/compressible-maxwell-from-zhong` | parameters, assembly/helper checks, docs | Gates experimental Citcom GMG and pressure diagnostics | Solver-diagnostic scope, not reference-state physics | Do not port |
| `7c301f6be3` | `codex/compressible-on-fix-zhong2022` | same patch as `ae4f615c3b` | Same gating on the correct base | Still depends on Citcom/GMG experimental path | Do not port |
| `3b4de78e8f` | private `codex/compressible-on-fix-zhong2022` ancestry | viscoelastic fields, mesh deformation, Citcom Stokes/GMG, tests | Adds scalar radial displacement, local stress ordering, displacement-only Citcom formulation, AMG/GMG comparison | `ve_radial_displacement` is not a full material-displacement vector; ALE mesh displacement is also sampled in volume terms | Do not port state or solver changes. Replace with one authoritative vector displacement |
| `54e93be01f` | private `codex/compressible-on-fix-zhong2022` | surface Love numbers, self-gravity MPI reductions | Collective-output correctness fixes | Unrelated to uniform constitutive integration; overlaps Phase-I self-gravity | Audit independently if a focused MPI failure requires it; do not port as compressible physics |

The prototype's `ve_pressure` is an advected compositional scalar with optional
bulk-viscous relaxation. It does not implement the required mixed pressure
history equation using ASPECT's pressure unknown. Its Citcom displacement-only
operator deliberately removes velocity-pressure coupling. Neither behavior is
the Phase-II baseline.

The reusable ideas are limited to explicit `K`/Lamé input, a material output
that supplies elastic bulk modulus, uniform compressible tests, and the already
present BE/theta/exponential deviatoric Maxwell updates. The stress-update
schemes are already in the Phase-I base and require no port.

## 3. Incompressible and compressible Maxwell paths

The current viscoelastic model is an incompressible material model. It advances
deviatoric stress history in existing stress compositional fields and returns
physical density from the incompressible multicomponent equation of state.
This path remains the default.

The uniform elastic-compressible path will add a default-off option to the
viscoelastic material model and expose an elastic bulk modulus `K`. It will not
turn thermodynamic compressibility into elastic compressibility and will not
make material density pressure-dependent. The pressure unknown itself is the
authoritative volumetric elastic pressure perturbation.

With positive pressure denoting compression,

```{math}
p'=-K\,\nabla\cdot\mathbf u,
```

and backward integration of

```{math}
\frac{d p'}{dt}=-K\,\nabla\cdot\mathbf v
```

gives

```{math}
p'^n-p'^{n-1}=-K\,\Delta t\,\nabla\cdot\mathbf v^n.
```

The mixed weak equation is

```{math}
(q,\nabla\cdot\mathbf v^n)
+\left(q,\frac{p'^n-p'^{n-1}}{K\Delta t}\right)=0.
```

ASPECT assembles the continuity row with the opposite overall sign, so the new
assembler adds a negative pressure-mass matrix and the matching negative old
pressure right-hand side. At timestep zero, `p'^{-1}=0` and the material
model's initial elastic interval replaces the zero simulator timestep.

Because this pressure mass term removes the constant-pressure nullspace, this
formulation requires `Pressure normalization = no`. It initially supports the
assembled AMG/direct solver paths; matrix-free GMG is rejected until the same
operator is implemented and compared independently.

## 4. Existing displacement-like quantities

| Quantity | Current ownership | Meaning | Why it is not the Phase-II material displacement |
| --- | --- | --- | --- |
| Mesh displacement | mesh-deformation handler | ALE node motion and interior mesh extension | Interior mesh velocity is chosen for mesh quality, not material motion |
| Surface/CMB deformation coefficients | potential-feedback plugins | Boundary-normal deformation used by sheet potentials | Boundary-only harmonic state, not a volume vector field |
| Surface Love-number displacement | postprocessor | Time-integrated boundary harmonic coefficients | Postprocessor-local and incomplete in the volume |
| Rotational-feedback predicted displacement | rotational-feedback plugin | Boundary radial old value plus one velocity increment | Specialized feedback state, not a general material field |
| Viscoelastic stress fields | material model/compositions | Advected deviatoric stress history | Stress is not displacement and does not determine rigid/solenoidal motion |
| Prototype `ve_radial_displacement` | rejected compressible branch | One advected radial scalar | Lacks tangential components and a coordinate-independent divergence |
| `solution`, `old_solution`, `old_old_solution` | simulator | Instantaneous velocity, pressure, temperature, composition states | Velocity history is available but no accumulated vector is stored |

No current quantity is an authoritative volume material displacement. Mesh and
material displacement coincide in boundary-normal velocity where the free
surface follows the material, but that limited equality does not justify using
the interior ALE extension for mass conservation.

## 5. Density consumers

The Phase-II mechanical perturbation applies to the same volume consumers
centralized in Phase I:

| Consumer | Required view |
| --- | --- |
| Standard, Newton, and anisotropic Stokes body force | selected Stokes source; mechanical law uses `delta_rho` |
| Internal self-gravity spherical-harmonic volume integral | `delta_rho` |
| Internal center-of-mass/degree-1 dipole | `delta_rho` |
| Geoid internal-density contribution | same self-gravity implementation |
| Perturbation-potential diagnostics | same self-gravity implementation |
| New density diagnostics/output | explicit reference, perturbation, and reconstructed views |

Energy, entropy, heating, thermal expansivity, compressibility, phase changes,
timestep estimates, and general material diagnostics continue to use physical
material density or their existing equation-specific reference choice.

## 6. State ownership and typed views

The ownership contract is:

- `rho_material`: produced by the material model and stored only in
  `MaterialModelOutputs::densities`;
- `rho_ref`: produced by the selected `DensitySourceManager` reference
  provider;
- `u_old`: committed accumulated material displacement owned by the manager;
- `u_trial = u_old + Delta t v`: evaluated during a Stokes/self-gravity
  iteration without mutating committed state;
- `delta_rho`: produced by the active source law from one explicit evaluation
  state;
- `rho_current = rho_ref + delta_rho`: reconstructed diagnostic/future-backend
  view, not a material-model output;
- `p'`: ASPECT's pressure unknown under elastic pressure evolution, positive in
  compression;
- `K`: elastic bulk modulus supplied through a typed additional material
  output, distinct from thermodynamic compressibility.

The manager API will use explicit names:

```cpp
physical_material_density(...)
reference_density(position)
reference_density_gradient(position)
current_displacement(...)
old_displacement(...)
displacement_divergence(...)
density_perturbation(...)
reconstructed_current_density(...)
stokes_source_density(...)
self_gravity_source_density(...)
```

The existing `physical_density()` identity accessor remains as a compatibility
alias during this phase and can be deprecated separately.

## 7. Authoritative displacement update

The manager stores a distributed finite-element vector with the same vector
components and polynomial space as ASPECT velocity. Only its velocity block is
meaningful. During a solve, consumers evaluate

```{math}
\mathbf u_\mathrm{trial}
=\mathbf u_\mathrm{committed}+\Delta t_\mathrm{effective}\mathbf v_\mathrm{linearization}.
```

The committed vector changes exactly once, after the nonlinear/time-stepping
manager accepts the timestep and before normal postprocessing. It is not
changed by individual Stokes or self-gravity iterations. If the timestep is
repeated, the committed vector is untouched. Timestep-zero instantaneous
elastic response uses the material model's initial elastic interval and is
committed only after initial mesh-refinement cycles are finished.

Postprocessors that run on nonlinear iterations are initially incompatible
with the mechanical law because there is no accepted state; this combination
will fail explicitly.

The first implementation will reject checkpoint resume and post-initialization
AMR with `linearized mass conservation`. Silent recomputation or interpolation
of an incomplete history is not allowed. Initial global refinement is safe;
initial adaptive refinement remains allowed only when solvers/postprocessors
are skipped until the final initial mesh, matching the frozen-profile rule.

## 8. Analytical radial reference provider

The new smooth provider is

```{math}
\rho_\mathrm{ref}(r)=\rho_0+\alpha(r-r_0),
```

where `alpha = d rho_ref / dr`. The spherical geometry model's natural
coordinate supplies `r`; the implementation rejects non-spherical natural
coordinates. The gradient is constructed from a purely radial vector in the
geometry's spherical coordinates and transformed to Cartesian components:

```{math}
\nabla\rho_\mathrm{ref}=\alpha\,\mathbf e_r.
```

Parameters are:

```text
subsection Formulation
  subsection Density sources
    set Reference density model = analytical radial
    subsection Analytical radial reference density
      set Reference density at reference radius = 0
      set Reference radius = 0
      set Radial density gradient = 0
    end
  end
end
```

Units are kg/m3, m, and kg/m4, respectively. Positive gradient means density
increases outward. The defaults are inactive because the default reference
model remains `none`. Hooks named `reference_pressure()` and
`reference_gravity()` may be added later, but they will throw `ExcNotImplemented`
until a physical analytical pressure/gravity state is specified.

## 9. Linearized mass-conservation law

The authoritative formula is

```{math}
\delta\rho=-\nabla\cdot(\rho_\mathrm{ref}\mathbf u)
=-\rho_\mathrm{ref}\nabla\cdot\mathbf u
-\mathbf u\cdot\nabla\rho_\mathrm{ref}.
```

It is evaluated from one explicit displacement state and never approximated by
`rho_material-rho_ref`. Coordinate-independent tensor products are used after
the reference provider returns its Cartesian gradient.

This produces the required limits:

- constant/incompressible: zero;
- constant/compressible: `-rho_ref div(u)`;
- radial/incompressible: `-u dot grad(rho_ref)`;
- radial/compressible: both terms.

Composition/thermochemical anomalies remain available through the legacy and
material-minus-reference laws. Phase II does not automatically add material
and mechanical anomalies, because an EOS may already contain the same density
change. Exactly one source law is active.

## 10. Pressure consistency diagnostic

For the elastic-pressure formulation,

```{math}
\delta\rho_p=\rho_\mathrm{ref}\frac{p'}{K}
-\mathbf u\cdot\nabla\rho_\mathrm{ref}.
```

The sign follows `p'=-K div(u)`: compression has negative divergence, positive
pressure, and positive constant-reference density perturbation. Pressure is a
diagnostic only; displacement remains authoritative.

The diagnostic reports L2 norms, maximum differences, and

```{math}
R_{\rho p}=\frac{\|\delta\rho_u-\delta\rho_p\|_2}
{\|\delta\rho_u\|_2+\epsilon}.
```

Elastic pressure evolution disables pressure normalization. Other formulations
retain their historical normalization. The diagnostic will report raw values;
it will not silently remove a constant pressure offset.

## 11. Boundary sources and total-ready views

The smooth radial volume perturbation and boundary sheets remain distinct:

```text
volume:  -rho_ref div(u) - u dot grad(rho_ref)
surface/CMB sheet: Delta rho * u_n
```

With linear interpolation, the current mechanical law differentiates the
selected tabulated radial profile piecewise. A sharp jump may instead be owned
by an explicitly tracked sheet, in which case its narrow table interval is
excluded from the volume gradient. With `Tabulated reference density
interpolation = piecewise constant`, every interior table radius automatically
owns the adjacent density difference as a sheet and the within-layer volume
gradient is zero. Composition-defined internal contrasts remain volume
material anomalies under the corresponding source law.

`reconstructed_current_density = rho_ref + delta_rho` is diagnostic and
prepares a future API. It does not enable total gravity. A total-potential
backend also needs core mass, exterior boundary conditions, moving-geometry
consistency, background subtraction, and volume/sheet de-duplication.

## 12. Parameters and compatibility

New choices are default-off:

```text
subsection Formulation
  set Mass conservation = elastic pressure evolution
  subsection Density sources
    set Reference density model = tabulated radial
    set Density source law      = mechanical mass conservation
    set Tabulated reference radii     = ...
    set Tabulated reference densities = ...
    set Tabulated reference density interpolation = piecewise constant
  end
end

subsection Material model
  subsection Viscoelastic
    set Enable compressible Maxwell      = false
    set Elastic bulk modulus formulation = bulk modulus
    set Elastic bulk moduli               = 2e11
    set Elastic Lame lambda moduli        = 1.5e11
  end
end
```

`Enable compressible Maxwell = false`, `Reference density model = none`, and
`Density source law = legacy` preserve old PRMs. The Lamé option computes
`K=lambda+2G/3`. Elastic pressure evolution requires elasticity, the
viscoelastic bulk output, assembled Stokes, and no pressure normalization.
Mechanical mass conservation additionally requires operator splitting and a
generic discontinuous field named `ve_radial_displacement`.
Thermodynamic `isentropic compression` remains a separate existing option.

## 13. Tests and acceptance criteria

The implementation gates are:

1. Existing Phase-I legacy, frozen, zero, and geoid tests remain unchanged.
2. A uniform compressible test verifies the discrete pressure increment and
   the default-off incompressible path.
3. Constant-reference incompressible linearized density agrees with zero
   volume perturbation.
4. Constant-reference compressible density agrees with `rho_ref p'/K` under
   refinement.
5. Analytical-radial manufactured displacement tests separately exercise the
   gradient-only and gradient-plus-divergence formulas.
6. Stokes and self-gravity/COM/geoid accessors produce identical perturbations
   for the same evaluation state.
7. `rho_current-rho_ref-delta_rho` is roundoff zero.
8. A no-load model remains stationary.
9. Restart and dynamic-AMR combinations fail with documented messages until
   history serialization/transfer is implemented.
10. Existing Zhong2022 incompressible sheet/FSSA runs remain unchanged in
    legacy mode and match zero-volume linearized results where applicable.

Timestep histories reaching the same physical time will be compared for
temporal convergence. Manufactured tests will report observed spatial order;
no convergence claim will be made from a single mesh.

## 14. Commit sequence

1. `integration: add elastic pressure evolution to viscoelastic Maxwell`
   selectively adds uniform `K`, the mixed assembler, and default/no-op tests.
2. `feat: add analytical radial reference-density provider` adds density,
   gradient, parameters, and provider tests.
3. `feat: centralize accumulated material displacement` adds trial/commit
   state and explicit restart/AMR limitations.
4. `feat: add linearized mass-conservation density source` implements the
   mechanical formula and reconstructed density.
5. `refactor: route mechanical density consumers through the manager` makes
   Stokes, self-gravity, COM, and geoid share the same evaluated state.
6. `feat: add pressure-density consistency diagnostics` adds raw norms and
   sign-convention tests.
7. `test: add reference-state mass-conservation suite` adds manufactured,
   no-load, timestep, reconstruction, and Zhong2022 comparisons.
8. `docs: document reference-state density and displacement ownership` updates
   user-facing methods and final limitations.

Each implementation commit must build independently. A commit may combine two
adjacent items only if separating them would leave an unbuildable public API.

## 15. Deferred work

Phase III will add a tabulated radial provider, PREM density/gravity/pressure,
radial elastic moduli, VM5a viscosity, and the Yuan2025 `(l,m)=(2,0)`
perturbation benchmark. It will also implement restart serialization and AMR
transfer before those features are enabled with long histories.

Phase T0 remains a separate later prototype. It will solve total potential on
fixed geometry and compare

```{math}
\Phi_\mathrm{total}[\rho_\mathrm{ref}+\delta\rho]
-\Phi_\mathrm{total}[\rho_\mathrm{ref}]
```

against the perturbation potential from `delta_rho`. This branch does not add
a total-gravity parameter, total body force, PREM, or altered FSSA/sheet
semantics.
