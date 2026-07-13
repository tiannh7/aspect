(sec:methods:volume-density-sources)=
# Volume-density sources

ASPECT material models return physical material density in
`MaterialModelOutputs::densities`. The central density-source framework does
not change this contract and does not require a material model to compute a
density perturbation. Instead, it selects the density consumed by Stokes
momentum and internal self-gravity volume calculations after the material model
has been evaluated.

The framework is configured under `Formulation/Density sources`. The reference
density model and density-source law are independent choices.

## Reference-density models

`Reference density model` has the following values:

- `none` (default) defines no central reference and returns zero when the
  reference-density accessor is queried;
- `constant` returns `Constant reference density`, in kg/m3;
- `frozen initial lateral average` computes a reference profile once from the
  initial physical material density.

The frozen model is evaluated after initial temperature, composition, and
pressure have been initialized on the final initial mesh, and before the first
Stokes or self-gravity volume calculation. It divides geometric depth into
`Frozen reference density profile slices`, computes quadrature-weighted lateral
averages with MPI reductions, stores the values at bin centers, and later uses
linear interpolation with endpoint clamping:

```{math}
\rho_\mathrm{ref}(z)
=
\left<\rho_\mathrm{material}(\mathbf x,t=0)\right>_z.
```

The profile remains frozen for the run. Initialization prints its integrated
anomaly, L2 norm, maximum anomaly, and maximum depth-bin lateral-mean residual.

Analytical radial and tabulated radial reference models are planned but are not
implemented.

## Density-source laws

`Density source law` has the following values:

- `legacy` (default) preserves historical consumer-specific behavior. Stokes
  momentum uses physical material density. Internal self-gravity potential and
  centre-of-mass/degree-1 integrals use physical density minus their existing
  `Reference density for internal anomalies` value;
- `material density` uses physical material density as the volume source;
- `material minus reference` uses

  ```{math}
  \rho_\mathrm{source}(\mathbf x,t)
  =\rho_\mathrm{material}(\mathbf x,t)
  -\rho_\mathrm{ref}(\mathbf x);
  ```

  for both Stokes momentum and internal self-gravity/degree-1 volume
  calculations;
- `zero volume perturbation` returns zero for those volume sources.

The geoid internal volume contribution reuses the internal self-gravity
calculation and consequently uses the same selected source. The
`material density` option does not implement total-field self-gravity: the
existing potential-feedback implementation still evaluates only its configured
spherical-harmonic perturbation degrees, and the background gravity model is
unchanged.

The default combination is `none + legacy`. Old parameter files therefore need
no edits. Existing dynamic-pressure Stokes parameters and the existing internal
self-gravity reference-density parameter retain their old meanings in legacy
mode. They are not silently mapped to the new parameters. A non-legacy law
cannot be combined with dynamic-pressure reference subtraction because that
would subtract two independent reference forces.

## Volume and sheet ownership

Composition-dependent density remains part of physical material density. An
irregular internal interface such as a composition-defined Moho therefore
remains a volumetric density anomaly and is automatically included in a
selected nonzero volume-source law.

An explicitly tracked sharp interface may instead be represented by a sheet
source in a future implementation. The same physical interface must never be
represented simultaneously as both a material/composition volume anomaly and
an explicit sheet source.

Existing surface-load, surface-topography, CMB-topography, and external-load
sheet sources are unchanged. Surface and CMB restoring terms, including the
Zhong2022-style local radial restoring term, continue to use their explicit
interface density jumps and are also unchanged.

## Other density consumers

Energy and entropy equations, heating models, timestep calculations, material
and heat diagnostics, generic gravity-point outputs, and visualization density
anomalies retain their existing physical or equation-specific reference-density
semantics. They are not routed through the volume-source law.

ASPECT nullspace removal is likewise unchanged. It removes numerical rigid
translations or rotations from velocity. It is distinct from physical
mass-redistribution, degree-1, or polar-wander feedback.

## Zhong2022 examples

A frozen initial reference state with a shared volume-density anomaly uses:

```text
subsection Formulation
  subsection Density sources
    set Reference density model                 = frozen initial lateral average
    set Frozen reference density profile slices = 100
    set Density source law                      = material minus reference
  end
end
```

A uniform incompressible case that retains only the existing surface/CMB sheet
and restoring terms can use:

```text
subsection Formulation
  subsection Density sources
    set Reference density model    = constant
    set Constant reference density = 4000
    set Density source law         = zero volume perturbation
  end
end
```

The constant value documents the reference state in this combination; the zero
law makes the volume source identically zero.

## Limitations

- Frozen reference profiles do not yet support pressure-dependent material
  density because a thermodynamic reference-pressure policy is not defined.
- Frozen profiles are not serialized for checkpoint/restart.
- Non-legacy laws are not yet supported with melt transport.
- Analytical/tabulated radial profiles, PREM/VM5a, material-provided anomalies,
  tracked internal interface sheets, and linearized mass conservation

  ```{math}
  \delta\rho=-\nabla\cdot(\rho_\mathrm{ref}\mathbf u)
  ```

  are future work.
