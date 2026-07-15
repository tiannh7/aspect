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

When `Mesh deformation/Use displacement history in free surface stabilization`
is enabled, the free-surface stabilization assembler supplies the committed
local restoring traction $\Delta\rho g h$ on each stabilized boundary. The
self-gravity boundary traction then applies only the non-local potential term,
so the committed displacement is not counted twice. The same committed relief
still enters the self-gravity potential calculation, and the stabilization
matrix still treats the displacement increment implicitly. The stabilization
density contrasts should therefore match the corresponding interface densities
under `Potential feedback/Interface properties`.

With `Formulation/Density sources/Density source law = mechanical mass
conservation`, radial internal density jumps contribute the sheet mass `Delta
rho U_r` to both the internal self-gravity spherical-harmonic integral and the
degree-one mass dipole. Jumps may be explicitly configured or derived from a
piecewise-constant tabulated reference state. These terms are active only when
`Include internal density anomalies` resolves to true and do not replace the
separate surface or CMB relief terms. Table-derived interfaces are identified
from the adjacent inner and outer radial cells in both the potential and
degree-one integrations, so ALE motion does not detach them from their
reference radii. Explicitly configured jumps retain radius-based matching with
the configured face tolerance.

For the 3-D `mechanical mass conservation` density-source law, self-gravity is
not a boundary-only force. ASPECT caches the current mass-potential
spherical-harmonic coefficients on the tabulated radial reference points and
linearly interpolates them in radius. In addition to the existing surface and
CMB potential tractions, the Stokes weak form contains

```{math}
-\int_\Omega \rho_\mathrm{ref}\Phi\,\nabla\cdot\mathbf w\,dV
+\sum_{\Gamma_\mathrm{internal}}
\int_\Gamma \Delta\rho\Phi
(\mathbf w\cdot\mathbf e_r)\,dS.
```

Together these terms reproduce the full-domain compressible potential forcing
used by CitcomSVE. The cached mass potential contains the external surface
load, surface and CMB deformation, the mechanical volume-density perturbation,
and internal density sheets. Tidal, rotational, and reference-frame potentials
remain separate and are not included in this volume term.

`Potential feedback/Self gravity/Full domain volume source discretization`
controls how the mechanical volume-density perturbation enters this cache.
The default `quadrature point` value preserves the pointwise ASPECT integral.
The default-off `cell average` value replaces the source within each active
cell by its volume-weighted average before applying the spherical-harmonic
Green kernel. `radial layer midpoint` additionally uses an arithmetic
quadrature-point source average and evaluates the radial Green kernel and
radial measure at the midpoint of the cell's inner and outer vertex radii.
`mass lumped radial layer` first projects the arithmetic cell averages to the
shared Q1 pressure vertices with a lumped mass matrix, then interpolates that
nodal source while using the same radial midpoint rule. The mass-lumped mode
requires vertex-supported continuous pressure shape functions. These options
are useful for coarse cross-code comparisons with
element-averaged radial-layer formulations; they do not change the Stokes
finite-element pair or any interface-sheet source.

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

## Glacial isostatic adjustment

The `glacial isostatic adjustment` mechanism adds a coupled time-dependent ice
and ocean surface load. It is implemented inside the unified potential-feedback
adapter so that the traction applied to Stokes is also the mass source used by
self-gravity, the degree-one center-of-mass correction, and rotational
feedback. The feature is disabled by default and requires `self gravity` in the
same mechanism list.

Ice and prescribed ocean functions are read as structured longitude-colatitude
histories. `Data format = citcomsve regular grid` reads the canonical CitcomSVE
header and longitude-latitude values directly, converts degrees to ASPECT's
radian longitude-colatitude coordinates, and closes the grid periodically at
the longitude seam and continuously at the poles. The default
`aspect structured data` format retains ASPECT's standard `# POINTS:` input.
Their schedule may contain elapsed model time and file number, or use the
CitcomSVE stage-age format. The latter starts with the number of stage intervals
and then lists one more row of age in ka and CitcomSVE time-step count. ASPECT
uses the age column to construct elapsed stage times and numbers the data files
sequentially; the CitcomSVE time-step count does not override ASPECT's time-step
selection.

The implementation follows the single sea-level equation in the canonical
CitcomSVE 3.0 source:

```{math}
L=O(N-U+c),
```

Here `N` is the current geoid-height perturbation, `U` is radial solid-surface
displacement, and `O` is the prescribed current ocean function. The spatially
uniform constant is

```{math}
c=\frac{-\Delta M_i/\rho_w-\int (N-U)O\,dS}{\int O\,dS}.
```

The ice-mass change is current prescribed ice mass minus the selected reference
ice mass. Ice loss therefore has `Delta M_i < 0` and adds ocean water. Ocean
function values are read from a static or time-dependent history and
clipped to the interval from zero to one. Stage interpolation may therefore
produce fractional ocean coverage, matching the canonical CitcomSVE update.
Shoreline and grounding changes must be encoded in that history; ASPECT does
not introduce a second online moving-shoreline SLE.

The total applied surface mass anomaly is

```{math}
\sigma_{\mathrm{GIA}}=\Delta\sigma_i+\rho_w L,
```

and its traction is `-g sigma_GIA n`. The load is stored as spherical-harmonic
coefficients through `Glacial isostatic adjustment/Maximum degree`, which makes
the same field available on the Stokes mesh after refinement and in restart
files. The `sea level` postprocessor detects an active coupled GIA provider and
outputs its sea-level change, ocean function, ice load, ocean load, total load,
barystatic constant, and eustatic contribution.

A coupled GIA configuration has the following form:

```prm
subsection Boundary traction model
  set Prescribed traction boundary indicators = top: potential feedback, \
                                                bottom: potential feedback
end

subsection Potential feedback
  set List of feedback mechanisms = self gravity, \
                                    rotational feedback, \
                                    glacial isostatic adjustment

  subsection Glacial isostatic adjustment
    set Ice load reference = first history file
    set Maximum degree = 32

    subsection Ice history
      set Data directory = /path/to/gia-data/
      set Data file name = ice.%d.txt
      set Data format = citcomsve regular grid
      set Schedule file name = ice_stages.txt
      set Schedule format = citcomsve stage ages
    end

    subsection Prescribed ocean function history
      set Data directory = /path/to/gia-data/
      set Data file name = ocean.%d.txt
      set Data format = citcomsve regular grid
      set Schedule file name = ocean_stages.txt
      set Schedule format = citcomsve stage ages
    end
  end
end
```

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
