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
  initial physical material density;
- `tabulated radial` interpolates `Tabulated reference radii` and `Tabulated
  reference densities` in spherical radius, with endpoint clamping. The
  default `Tabulated reference density interpolation = linear` is continuous.
  The default-off `piecewise constant` mode uses the density at the lower
  radius in each outward interval.

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
- `zero volume perturbation` returns zero for those volume sources;
- `mechanical mass conservation` is available with `elastic pressure
  evolution`. It uses ASPECT's pressure as compressive elastic pressure and a
  generic discontinuous field named `ve_radial_displacement` as committed
  radial material-displacement history. The field may use the `field` method
  for a material-coordinate history, or the `static` method for a
  reference-mesh history. The latter still receives operator-splitting
  reaction updates but is not passed through the compositional advection
  solve:

  ```{math}
  \delta\rho
  = \frac{\rho_\mathrm{ref}}{K}p
  - U_r\frac{d\rho_\mathrm{ref}}{dr}.
  ```

  The Stokes body-force selector returns zero for this law because the matching
  local `rho g` terms are assembled directly in the elastic operator. Internal
  self-gravity and degree-1 volume integrals consume the reconstructed
  perturbation above.

  When self gravity is active in 3-D, the same volume and sheet perturbations
  also generate a potential cached on the radial reference points. The
  compressible Stokes feedback uses the matching volume and internal-interface
  weak terms; it is therefore not limited to surface and CMB tractions.

  These terms are implemented on the fine grid for assembled AMG/direct and
  local-smoothing `block GMG`: the latter includes the elastic pressure mass,
  mechanical radial cell couplings, and internal-density-jump face restoring
  operator. Its multigrid level operators remain simplified preconditioner
  approximations and may omit the mechanical radial and internal-face terms;
  this changes preconditioning, not the fine-grid linear system.
  Global-coarsening GMG remains explicitly rejected.

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

For a radial reference state, explicitly tracked sharp interfaces can be
configured with `Internal density jump radii`, `Internal density jump density
contrasts`, and `Internal density jump face tolerance`. These lists are empty
by default. They are currently available only with `mechanical mass
conservation` and a `tabulated radial` reference state, and every configured
radius must coincide with a radial mesh face. Contrasts are defined as density
below minus density above.

Alternatively, `Tabulated reference density interpolation = piecewise
constant` derives every internal contrast directly from adjacent table values.
For an interior table radius `r_i`, the contrast is
`rho_ref(r_i^-)-rho_ref(r_i^+)`. This mode requires `mechanical mass
conservation`, requires the table radii to coincide with radial mesh faces,
and requires the explicit jump lists to remain empty. It therefore gives the
volume and sheet terms one consistent radial discretization and owner.

Each interface contributes the local restoring operator

```{math}
\int_\Gamma \Delta\rho g\Delta t
(\mathbf w\cdot\mathbf e_r)(\mathbf v\cdot\mathbf e_r)\,dS
-\int_\Gamma \Delta\rho g
(\mathbf w\cdot\mathbf e_r)U_r^\mathrm{old}\,dS.
```

When internal density anomalies are included in potential feedback, the same
interface contributes sheet mass
`sigma = Delta rho U_r` to the self-gravity potential, degree-one mass dipole,
and rotational-feedback inertia tensor. Piecewise-constant table interfaces use
the same adjacent-cell identity for these non-local terms as for the local
restoring operator, and therefore remain active after ALE mesh deformation.
Explicit jumps retain the configured radius tolerance. A narrow tabulated
interval containing an explicitly configured jump is excluded from the volume
reference-density gradient, preventing double counting. The same physical
interface must not also be represented as a material/composition volume
anomaly.

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

A radial finite-bulk benchmark uses:

```text
subsection Formulation
  set Enable elasticity = true
  set Mass conservation = elastic pressure evolution
  subsection Density sources
    set Reference density model = tabulated radial
    set Density source law      = mechanical mass conservation
    set Tabulated reference radii     = ...
    set Tabulated reference densities = ...
    set Tabulated reference density interpolation = piecewise constant
    set Tabulated mechanical gravity magnitudes   = ...
    set Internal density jump face tolerance    = 1
  end
end

subsection Material model
  subsection Viscoelastic
    set Enable compressible Maxwell = true
    set Use ascii profile           = true
  end
end
```

This configuration also requires operator splitting, no pressure
normalization, spherical geometry, a generic discontinuous
`ve_radial_displacement` field, and either an assembled AMG/direct Stokes solver
or local-smoothing `block GMG`. Global-coarsening GMG remains rejected. It is
disabled unless explicitly selected and is distinct from thermodynamic
isentropic compression. Elastic Stokes right-hand-side assembly requests
viscosity explicitly so that viscoelastic material averaging is consistent
between assembled AMG and matrix-free GMG runs. Because elastic pressure
evolution has a finite pressure mass and no constant-pressure nullspace, its
pressure right-hand side is not compatibility-projected.

Use `Compositional field methods = ..., static` for
`ve_radial_displacement` when the reference density profile and its internal
jump sheets are fixed to the reference mesh. This reproduces a direct nodal
accumulation of radial displacement: operator splitting adds the radial
velocity increment, while no subsequent compositional transport is applied.
The legacy `field` method remains available for models that intentionally
transport the history with the material.

The optional `Tabulated mechanical gravity magnitudes` list contains one
constant magnitude per interval in `Tabulated reference radii`. When the list
is empty, which is the default, the local mechanical volume couplings use the
selected gravity model at each quadrature point. A nonempty list changes only
those volume couplings. Free-surface, CMB, and internal-interface restoring
terms still use the selected gravity model, so a layered benchmark can use
face/node gravity there while reproducing an element-centre volume
discretization. Values are in m/s^2.

Mechanical mass conservation requires `Initial response mode = instantaneous
elastic` at timestep zero by default. The default-false `Allow viscoelastic
initial mechanical response` parameter relaxes this check only when an
explicit time-integration discriminator needs the material model's finite
Maxwell response during the initial loaded solve. Later viscoelastic steps are
unchanged. Production benchmarks that define `t=0` as the instantaneous
elastic limit should leave this option disabled.

## Limitations

- Frozen reference profiles do not yet support pressure-dependent material
  density because a thermodynamic reference-pressure policy is not defined.
- Frozen profiles are not serialized for checkpoint/restart.
- Non-legacy laws are not yet supported with melt transport.
- The mechanical law currently supports constant or tabulated radial reference
  states and a scalar radial displacement history. A future general model needs
  an authoritative vector material displacement.
- Explicit internal density jumps require spherical, constant-radius,
  jump-aligned mesh faces. General non-radial tracked interfaces and
  composition-defined sheets are not implemented.
- PREM/VM5a inputs have not yet passed the required short G2 scientific
  comparison with canonical CitcomSVE 3.0. The available local-smoothing GMG
  implementation must not be described as production-ready until that
  comparison checks the target harmonic, leakage, surface/CMB displacement,
  and stress.
