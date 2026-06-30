# Rotational Feedback and Center-of-Mass Correction

This document describes the optional `rotational feedback` boundary traction
model for spherical-shell surface-loading benchmarks.

The feature is intended for benchmark-scale polar-wander experiments such as a
degree-2/order-1 surface load. It is not a full glacial-isostatic-adjustment
implementation and does not solve the sea-level equation, ocean loading, or a
time-dependent Liouville equation.

## Naming

The externally prescribed degree-2 potential used by self-gravitation is named
`tidal potential`. Older input files using subsection `Applied potential` are
accepted as a deprecated compatibility path, but new input files should use
`Tidal potential`.

The mass-redistribution feedback caused by a perturbed spin axis is named
`rotational feedback`. This is intentionally separate from ASPECT's
`Nullspace removal` parameters, which remove algebraic rigid velocity modes
after a Stokes solve and do not represent physical polar wander.

## Algorithm

The `rotational feedback` plugin:

1. collects the effective boundary mass from external surface traction and ALE
   topography;
2. optionally removes the degree-1 part of the effective surface mass as a
   center-of-mass frame correction;
3. computes the products of inertia `Delta Ixz` and `Delta Iyz`;
4. computes a small rotation-vector perturbation from
   `delta_omega_x = Omega * Delta Ixz / (C - A)` and
   `delta_omega_y = Omega * Delta Iyz / (C - A)`;
5. evaluates the induced centrifugal-potential perturbation
   `delta Phi = Omega * z * (delta_omega_x*x + delta_omega_y*y)`;
6. applies `delta Phi/g` as an additional normal traction at selected
   boundaries.

The center-of-mass correction in this first version is deliberately limited: it
removes degree-1 effective surface mass before computing the rotational inertia
perturbation and reports the apparent center-of-mass shift. It does not replace
a full GIA center-of-mass reference-frame treatment.

## Benchmark Usage

The comparison pair

- `prms/BM01_single_harmonic_surface_load/V1_uniform/l2m1/BM01-V1-l2m1-ref2-theta0p5-SH-polar-wander-off.prm`
- `prms/BM01_single_harmonic_surface_load/V1_uniform/l2m1/BM01-V1-l2m1-ref2-theta0p5-SH-polar-wander-on.prm`

keeps the same surface load and self-gravity settings, changing only the
rotational-feedback forcing. The l=2,m=0 benchmark should remain insensitive to
this option to first order, while l=2,m=1 should show a measurable degree-2
response change when the feature is enabled.
