# Vector material-displacement history

## Scope

This branch adds an optional diagnostic material-displacement vector to the
viscoelastic material model. It does not replace
`ve_radial_displacement`, change the mechanical density source, or change
the default surface Love-number output.

ASPECT compositional fields are scalar fields, so the vector is represented by
the Cartesian component fields

```text
ve_displacement_x
ve_displacement_y
ve_displacement_z
```

in three dimensions (and the first two fields in two dimensions). The fields
must either all be present or all be absent.

## Evolution equation

For material displacement \(\boldsymbol d(\boldsymbol x,t)\), material
velocity \(\boldsymbol u\), and ALE mesh velocity \(\boldsymbol w\), the
material equation is

\[
\left.\frac{\partial\boldsymbol d}{\partial t}\right|_{\rm mesh}
+(\boldsymbol u-\boldsymbol w)\cdot\nabla\boldsymbol d
=\boldsymbol u.
\]

The existing compositional advection assembler already subtracts the mesh
velocity when mesh deformation is active. The viscoelastic material model
therefore supplies the Cartesian components of \(\boldsymbol u\) as
operator-splitting reaction rates. With a full-Lagrangian mesh
\(\boldsymbol w=\boldsymbol u\), the advective term vanishes.

The instantaneous-elastic timestep-zero convention follows the existing radial
history: timestep zero is initialized separately, and the first ordinary
reaction update must not commit it a second time.

## Compatibility and activation

The feature is activated only by declaring every required field with the exact
names above. Each field must be generic, discontinuous, and use either the
`field` or `static` compositional method.

`field` is the physical ALE choice because ASPECT then transports the history
with \(\boldsymbol u-\boldsymbol w\). `static` is allowed for
full-Lagrangian or reference-mesh diagnostics, where ordinary compositional
advection is intentionally disabled.

The existing scalar radial history remains the only history consumed by the
mechanical mass-conservation density source. During the diagnostic stage the
vector fields are independent outputs, preventing double counting.

## Derived quantities

Given a material reference direction
\(\boldsymbol e_{r,0}\), the vector provides

\[
d_r=\boldsymbol d\cdot\boldsymbol e_{r,0},\qquad
\boldsymbol d_t=\boldsymbol d-d_r\boldsymbol e_{r,0}.
\]

For the first implementation, tests use geometries where the reference radial
direction is known exactly. Production use must obtain this direction from a
material reference position rather than silently using a deformed current
position.

For a laterally varying reference density, the linearized Eulerian density
perturbation is

\[
\rho'=-\nabla\mathbin{\cdot}(\rho_0\boldsymbol d)
     =-\rho_0\nabla\mathbin{\cdot}\boldsymbol d
      -\boldsymbol d\mathbin{\cdot}\nabla\rho_0.
\]

With pressure positive in compression,
\(-\rho_0\nabla\mathbin{\cdot}\boldsymbol d=\rho_0p/K\). In spherical
components the material-advection term is

\[
-\boldsymbol d\mathbin{\cdot}\nabla\rho_0
\begin{aligned}
={}&-d_r\partial_r\rho_0
 -\frac{d_\theta}{r}\partial_\theta\rho_0\\
 &-\frac{d_\phi}{r\sin\theta}\partial_\phi\rho_0.
\end{aligned}
\]

Consequently, tangential displacement is physically required as soon as the
reference density or another material property varies laterally. Across a
sharp material interface, the corresponding sheet term depends on the normal
displacement \(\boldsymbol d\mathbin{\cdot}\boldsymbol n_\Sigma\), not
necessarily on the spherical radial component.

The finite-deformation form is most naturally written through the material
map \(\boldsymbol x=\boldsymbol\chi(\boldsymbol X,t)\):

\[
\rho(\boldsymbol x,t)
=\frac{\rho_0(\boldsymbol X)}
       {\det(\partial\boldsymbol x/\partial\boldsymbol X)}.
\]

This is the long-term reason to retain the full Cartesian displacement (or an
equivalent inverse material map), even though the radial PREM benchmark only
uses its radial projection.

## Independent surface diagnostic

When all three Cartesian fields are present, the surface Love-number
postprocessor independently projects their radial and poloidal components.
It writes `h_vector` and `l_vector` beside the existing
velocity-time-integrated `h_accumulated` and `l_accumulated` values. During
this diagnostic stage both paths intentionally use the same current-surface
basis and area weight, so their difference isolates displacement-history
transport and time integration rather than reference-surface mapping.

The time layers are deliberately not hidden. ASPECT commits compositional
fields before the current Stokes solve, whereas the legacy postprocessor
accumulator immediately adds the current post-Stokes velocity multiplied by
the timestep. They agree at the separately initialized instantaneous-elastic
timestep zero. At later timesteps their difference includes this current-step
offset as well as any genuine transport or integration error. Timestep
refinement is therefore required before changing the production displacement
source.

For the current solve ordering, the two updates are schematically

\[
\boldsymbol d_{\mathrm{vector}}^n
=\mathcal T_n(\boldsymbol d_{\mathrm{vector}}^{n-1})
 \Delta t_n\boldsymbol u^{n-1},
\qquad
\boldsymbol d_{\mathrm{accumulated}}^n
=\boldsymbol d_{\mathrm{accumulated}}^{n-1}
 \Delta t_n\boldsymbol u^n,
\]

where \(\mathcal T_n\) is compositional transport. The first ordinary vector
reaction after instantaneous-elastic initialization is skipped so that the
artificial elastic velocity is not committed twice. Consequently, with a
constant timestep and negligible projection/transport error, the committed
vector coefficient at step \(n\) matches the legacy accumulator at step
\(n-1\), not at step \(n\). Local 4x and 8x PREM tests confirm this relation.

With a changing timestep, the delayed vector update multiplies the previous
velocity by the current timestep. It can therefore no longer be compared to a
simple one-step shift of the legacy accumulator. This is a time-integration
design issue, not evidence that the Cartesian projection is inaccurate. A
production takeover must first define one committed displacement time layer
and use the interval length belonging to that velocity increment.

## Acceptance tests

1. A prescribed Cartesian velocity must update every displacement component
   by the expected amount through operator splitting.
2. A radial manufactured velocity must give
   \(\boldsymbol d\cdot\boldsymbol e_r =
   \texttt{ve_radial_displacement}\) to interpolation tolerance.
3. With `Mesh velocity formulation = material velocity`, the full-Lagrangian
   test must show that vector displacement updates without relative advection.
4. Checkpoint/restart must preserve all component fields.
5. A later surface diagnostic must compare the tangential spherical-harmonic
   coefficient computed from \(\boldsymbol d_t\) with the existing cumulative
   surface-velocity coefficient before the vector history is allowed to drive
   production Love numbers.
