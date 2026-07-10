(sec:methods:potential-feedback)=
# Potential feedback

The `potential feedback` boundary traction model applies normal tractions that
come from perturbations of gravitational or centrifugal potentials.  It is used
by self-gravity, tidal-potential, and rotational-feedback benchmark setups.
These terms are physical feedbacks and are separate from the numerical
nullspace removal described in {ref}`sec:methods:nullspace-removal`.

Self-gravity derives the total planet mass and mean density internally from the
spherical-shell outer radius and the surface gravity magnitude. Benchmark input
files therefore do not need separate planet-mass, mean-density, or moment-of-inertia
parameters.

Potential feedback always uses the current total relief of each active density
interface relative to the reference spherical interface. For a future model with
initial topography, the source height is therefore the current radius minus the
reference radius, including both the initial relief and any ALE displacement.

The active feedback interfaces are inferred from the boundaries on which the
`potential feedback` boundary traction plugin is prescribed in
`Boundary traction model/Prescribed traction boundary indicators`. Applying
`potential feedback` on the top/surface/outer boundary enables the surface
source and applies the resulting potential traction there. Applying it on the
bottom/CMB/inner boundary enables the CMB source and applies the resulting
potential traction there. On an active feedback boundary, any other boundary
traction plugin on the same boundary is treated as an externally applied load.

The self-gravity calculation starts internally at spherical-harmonic degree 1.
CitcomSVE 3.0 expands output arrays from degree 0 through `output_ll_max`, but
its self-gravity potential synthesis skips the degree-0 term. ASPECT follows the
same convention for the feedback calculation and exposes only the calculation
`Maximum degree` in `Potential feedback/Self gravity`.

Polar-wander rotational feedback is configured under
`Potential feedback/Rotational feedback`. Its `Fluid Love number` parameter is
the degree-2 fluid Love number $k_f$ used in the linearized polar-wander
relation. This is the same quantity as CitcomSVE's `polar_wander_kf`. CitcomSVE
3.0 computes the polar-wander feedback from the $I_{13}$ and $I_{23}$ products
of inertia and adds only the degree-2, order-1 centrifugal-potential
perturbation. ASPECT therefore fixes the rotational-feedback transform to this
degree and order in the unified `Potential feedback` path. With
$dI_{xz}=-\int dm\,xz$ and $dI_{yz}=-\int dm\,yz$, the perturbation is
$\Phi_r= -3G(dI_{xz}x+dI_{yz}y)z/(k_f R^5)$. Users therefore do not need to
provide separate polar and equatorial moments of inertia. The surface
coefficient of this centrifugal potential is included in surface-Love-number
geoid output, but it is not counted as a surface mass-potential contribution.

Rotational feedback is disabled unless `rotational feedback` is listed in
`Potential feedback/List of feedback mechanisms` and the `potential feedback`
boundary traction model is applied on the relevant boundary.

A CitcomSVE-style benchmark input uses the shared boundary-traction list for the
interfaces and keeps the feedback block compact:

```prm
subsection Boundary traction model
  set Prescribed traction boundary indicators = outer: spherical harmonic load, \
                                                outer: potential feedback, \
                                                inner: potential feedback
end

subsection Potential feedback
  set List of feedback mechanisms = self gravity, rotational feedback

  subsection Rotational feedback
    set Fluid Love number = 1.11664062
  end

  subsection Self gravity
    set Include internal density anomalies = false
    set Maximum degree = 32
  end
end
```

The `Surface love numbers` postprocessor remains an output filter. Its minimum
degree defaults to 0 so that coefficient output follows CitcomSVE 3.0's
degree-0-through-`output_ll_max` output convention unless the user chooses a
narrower diagnostic range.
