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
