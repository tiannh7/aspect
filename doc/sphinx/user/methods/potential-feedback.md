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

Polar-wander rotational feedback is configured under
`Potential feedback/Rotational feedback`.  Its `Fluid Love number` parameter is
the degree-2 fluid Love number $k_f$ used in the linearized polar-wander
relation.  This is the same quantity as CitcomSVE's `polar_wander_kf`.  The
implementation computes the products-of-inertia perturbation from the same
density interfaces listed in `Potential feedback/Self gravity/Boundary
indicators` and converts it directly to the degree-2 order-1
centrifugal-potential perturbation using $k_f$. With
$dI_{xz}=-\int dm\,xz$ and $dI_{yz}=-\int dm\,yz$, the perturbation is
$\Phi_r= -3G(dI_{xz}x+dI_{yz}y)z/(k_f R^5)$. Users therefore do not need to
provide separate polar and equatorial moments of inertia. The surface
coefficient of this centrifugal potential is included in surface-Love-number
geoid output, but it is not counted as a surface mass-potential contribution.

Rotational feedback inherits its source interfaces and traction boundaries from
`Potential feedback/Self gravity/Boundary indicators`: `outer`, `top`, and
`surface` select the surface contribution and outer-boundary traction, while
`inner`, `bottom`, and `CMB` select the CMB contribution and inner-boundary
traction.

Rotational feedback is disabled unless `rotational feedback` is listed in
`Potential feedback/List of feedback mechanisms` and the `potential feedback`
boundary traction model is applied on the relevant boundary.
