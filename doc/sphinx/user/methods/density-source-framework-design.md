(sec:methods:density-source-framework-design)=
# Unified density-source framework design

This report records the density-source audit and implementation design for the
Zhong et al. (2022) self-gravity work. It is intentionally written before the
C++ implementation. The validated base is commit `a4ea2d441d` on
`fix/zhong2022-cmb-local-topography-mode`. The useful implementation reference
is commit `9ab79022a2` on `feature/frozen-initial-reference-density`.

## 1. Current density-production paths

`MaterialModel::MaterialModelOutputs::densities` is the single general material
density output. Every active material model fills it from its equation of state,
temperature, pressure, composition, position, or a combination of these. Its
meaning is the physical material density at the evaluation point. This contract
must not change, and existing material models must not be required to return a
perturbation density.

Composition-defined density contrasts already enter through this path. For
example, the simple and viscoelastic material models obtain phase volume
fractions from compositional fields, evaluate phase densities, and write their
mixture to `MaterialModelOutputs::densities`. An irregular composition-defined
Moho is consequently a volumetric material-density contrast.

ASPECT also has adiabatic/reference-density profiles. They are deliberately
substituted for the physical material density in selected energy-equation,
stabilization, and timestep formulations. These are equation-specific
approximations and are not perturbation-density sources.

## 2. Density consumers

### Stokes momentum body force

The standard Stokes assembler in `source/simulator/assemblers/stokes.cc`, the
Newton residual assembler in `source/simulator/assemblers/newton_stokes.cc`, and
the anisotropic-viscosity assembler in
`source/simulator/assemblers/stokes_anisotropic_viscosity.cc` all currently use
`MaterialModelOutputs::densities[q] * gravity` directly. These three consumers
must use `DensitySourceManager::stokes_source_density()`.

Other density uses in the Stokes assemblers belong to compressible mass
conservation, stabilization, or Newton derivatives. They must continue to use
physical material density and are not density-source-law consumers.

The existing dynamic-pressure option subtracts a constant reference-density
body force through `AdditionalMaterialOutputsStokesRHS`. On the validated base,
the viscoelastic material model supplies that additional force. This behavior
must remain unchanged in legacy mode. A non-legacy central density-source law
must be incompatible with the old dynamic-pressure subtraction so that the
reference density cannot be subtracted twice.

The initial-lithostatic-pressure boundary traction has its own explicit
total/dynamic-pressure conversion. It is a boundary-traction construction, not
one of the volume body-force consumers being centralized here, and remains
unchanged in this phase.

### Internal self-gravity and degree one

`PotentialFeedback::SelfGravitation::compute_internal_density_potential()`
forms spherical-harmonic coefficients from

```{math}
\rho_\mathrm{material}-\rho_\mathrm{legacy\ selfgrav\ reference}.
```

`compute_internal_density_mass_dipole()` uses the same expression for the
internal contribution to the native centre-of-mass/degree-1 diagnostic. Both
the automatic zero-anomaly detector and the final integral must call
`DensitySourceManager::self_gravity_source_density()` so they cannot disagree.

The geoid postprocessor obtains its volume-density coefficients through the
same `SelfGravitation::compute_internal_density_potential()` implementation,
either from the active potential-feedback object or from its helper object.
Routing that method therefore also routes the applicable geoid/potential
diagnostic without a second implementation.

Surface-load, surface-topography, CMB-topography, and external-load terms are
explicit sheet sources. The surface and CMB radial restoring terms use their
specified interface density jumps. None of these terms is a volume-density
consumer and none will be changed.

### Diagnostics that retain their current semantics

The following consumers must not be routed through the perturbation source law:

- energy and entropy advection assemblers use physical density or their
  explicitly selected adiabatic/reference-density approximation;
- heating models use physical density to convert specific to volumetric heat;
- conduction and convection timestep calculations use physical density or the
  selected adiabatic reference profile;
- material, mass, heat-flux, boundary-density, dynamic-topography, mesh
  refinement, and lateral-average diagnostics retain their documented physical
  density semantics;
- the visualization `density anomaly` postprocessor retains its own selectable
  reference-profile/lateral-average definition;
- `Gravity point values` continues to output both total gravity from physical
  density and gravity anomaly relative to its own `Reference density`
  parameter;
- nullspace removal continues to use its current physical-density or constant
  weighting. Nullspace removal is a numerical reference-frame operation and is
  not physical polar-wander or density-source feedback.

## 3. Exact current semantics

| Consumer | Validated-base density | New legacy result |
| --- | --- | --- |
| Standard, Newton, anisotropic Stokes momentum | physical material density | unchanged |
| Dynamic-pressure additional Stokes RHS | minus configured constant, where implemented | unchanged |
| Internal self-gravity potential | material minus legacy self-gravity reference | unchanged |
| Internal COM/degree-1 dipole | material minus legacy self-gravity reference | unchanged |
| Geoid internal volume term | same self-gravity implementation | unchanged |
| Surface/CMB potential sheets | configured interface density jump | unchanged |
| Surface/CMB restoring terms | configured interface density jump | unchanged |
| Energy, heating, timestep | physical or equation-specific reference density | unchanged |
| Generic material/gravity diagnostics | their own documented physical/reference choice | unchanged |

## 4. Class and ownership design

The simulator will own one `DensitySourceManager<dim>`, exposed read-only
through `SimulatorAccess<dim>`. Material models remain owners of physical
density production. The manager owns only reference-density state and source
selection.

The public API uses typed names rather than an ambiguous `density()`:

```cpp
physical_density(outputs, q)
reference_density(position)
stokes_source_density(inputs, outputs, q)
self_gravity_source_density(inputs, outputs, q,
                            legacy_reference_density)
```

`physical_density()` is an explicit identity accessor that documents the
material-output contract. `reference_density()` selects the configured
reference model. `stokes_source_density()` and
`self_gravity_source_density()` select the configured law and preserve their
different historical behavior in legacy mode. The legacy self-gravity scalar
is an argument because it remains owned by the existing self-gravity settings.

The reference model and density-source law are independent enums. Future law
implementations can add displacement, reference-density gradients, or
material-provided auxiliary outputs inside the manager without changing every
consumer.

## 5. Parameters and defaults

The new layout is:

```text
subsection Formulation
  subsection Density sources
    set Reference density model = none
    set Constant reference density = 0
    set Frozen reference density profile slices = 100
    set Density source law = legacy
  end
end
```

Implemented reference models are `none`, `constant`, and
`frozen initial lateral average`. `analytical radial` and `tabulated radial`
are reserved for Phase II and are not accepted parameter values in this phase.

Implemented laws are `legacy`, `material density`, `material minus reference`,
and `zero volume perturbation`. `linearized mass conservation` and
`material provided` are reserved for future extensions and are not accepted
parameter values in this phase.

The default `none + legacy` preserves old parameter files exactly. The useful
new combinations are:

- `frozen initial lateral average + material minus reference` for a shared
  initial reference state;
- `constant + material minus reference` for a constant perturbation reference;
- `constant + zero volume perturbation` for a Zhong2022 incompressible volume
  source with only the existing boundary sheets/restoring terms retained.

`material minus reference` requires a non-`none` reference model. Existing
`Formulation/Stokes pressure` parameters and existing self-gravity reference
parameters remain supported and retain their meaning in legacy mode. They are
not silently mapped to the new parameters. Non-legacy laws reject the old
dynamic-pressure subtraction to prevent double counting. Explicitly selecting
a non-legacy law makes the old self-gravity scalar irrelevant by user choice;
the new documentation states this directly.

## 6. Composition compatibility

The manager receives already evaluated material-model inputs and outputs. It
never asks a material model to return a perturbation and does not modify
`outputs.densities`. Therefore a composition-defined density contrast remains
part of physical density. Under `material minus reference`, the same
composition-dependent physical density is used in both Stokes and internal
self-gravity before the same manager subtracts the same reference state.

A focused test will use a composition-dependent material density and verify:

1. physical density retains the composition contribution;
2. legacy Stokes receives exactly that physical density;
3. legacy self-gravity retains its historical scalar subtraction;
4. material-minus-reference Stokes and self-gravity return numerically
   identical volume-density sources.

No sheet source is introduced for the composition interface.

## 7. Volume versus sheet ownership

An internal contrast may be represented as a volumetric composition/material
anomaly. A future explicitly tracked sharp interface may instead be represented
as a sheet source. The same physical interface must never be represented by
both paths. This task keeps irregular internal interfaces on the existing
composition/material path and changes no surface or CMB sheet implementation.

## 8. Backward-compatibility risks

The main risks and controls are:

- missed Stokes variants: route standard, Newton, and anisotropic momentum
  assemblers, while leaving compressibility terms physical;
- changed legacy self-gravity: pass its existing scalar reference explicitly
  to the manager and compare old/new focused outputs;
- double reference subtraction: reject non-legacy laws with the existing
  dynamic-pressure mode;
- early or repeated frozen-profile initialization: initialize once after the
  final initial state and initial pressure are set, before the first consumer;
- pressure-dependent reference construction: reject frozen-profile mode until
  a thermodynamic pressure policy exists;
- restart corruption: reject non-legacy restart until reference state is
  serialized;
- interface double counting: document and test the volume-versus-sheet
  ownership rule;
- accidental energy changes: do not route energy/heating/timestep consumers.

## 9. Tests and numerical acceptance

Focused framework tests will cover default legacy selection, constant reference,
zero volume source, frozen-profile initialization/interpolation/diagnostics,
composition-dependent physical density, and identical non-legacy Stokes and
self-gravity source values. Existing Stokes, Newton, anisotropic, geoid, and
viscoelastic tests provide consumer-path regression coverage.

Acceptance requires:

- old parameter files parse without edits;
- legacy focused outputs match `a4ea2d441d` within the existing test tolerance;
- the uniform incompressible and compressible Zhong2022 Love-number outputs do
  not change in legacy mode;
- surface/CMB potential and restoring outputs do not change;
- composition-dependent Stokes forcing remains active;
- non-legacy Stokes, self-gravity potential, and COM/degree-1 use the same
  manager result;
- no energy/material-output semantics change;
- formatting, compilation, and focused test commands and results are recorded.

## 10. Commit split

Commit 1, `refactor: centralize density-source selection`:

- add the manager and typed legacy accessors;
- preserve `none + legacy` behavior;
- route standard, Newton, anisotropic Stokes and internal self-gravity/COM;
- add legacy and composition regression tests;
- include this audit/design report.

Commit 2, `feat: separate frozen reference-density model from anomaly law`:

- add independent reference-model and source-law parameters;
- add constant, material-density, material-minus-reference, and zero laws;
- port and refactor frozen initial lateral averaging, MPI reductions, and
  diagnostics from `9ab79022a2`;
- add non-legacy tests and user documentation.

Phase II will add analytical and tabulated radial providers, including PREM,
then implement

```{math}
\delta\rho=-\nabla\cdot(\rho_\mathrm{ref}\mathbf u)
```

inside a new `linearized mass conservation` law. It will require an explicit
displacement representation, radial-reference gradients, pressure policy,
restart serialization, and focused conservation tests. Phase II is not part of
this implementation.
