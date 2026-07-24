# Follow-up review of `f7fdd6eac`

Branch: `codex/degree-one-com-constraint`

Reviewed commit: `f7fdd6eac0cbdbd498e2485c1ccc16b5ac885a40`

## Decision

`f7fdd6eac` fixes the catastrophic normalization error in the former synthetic test, but the native coupled COM implementation is still not ready for benchmark acceptance without the follow-up patch stored in:

```text
doc/review/degree_one_com_constraint_followup_f7_v2.patch
```

## What `f7fdd6eac` fixed

The old test combined `g=1`, `R=2`, and order-one densities, so `gR^2/G` exceeded the represented model mass by about 1.6 billion. The new test chooses a gravity consistent with the synthetic model mass and therefore exercises a finite COM correction instead of a numerically negligible one.

## Remaining blocking findings

### 1. COM mass currently changes every self-gravity degree kernel

The commit assigns its density-integrated mass to `planet_mass`, then derives `planet_mean_density` from that value. Surface and CMB Green kernels use `planet_mean_density` for every degree. Consequently enabling the COM frame changes the complete self-gravity operator, including degrees two and above.

The COM constraint mass and the Green-kernel normalization must be separate variables. The kernels retain the mass implied by `g(R)R^2/G`; only `D-Mc=0` uses the explicit COM reference mass.

### 2. The density immediately below the CMB is not a core mean density

The new helper computes unresolved core mass as

$$
M_c=\frac{4\pi}{3}\rho_{\mathrm{below\ CMB}}R_c^3.
$$

For PREM/VM5a or Mars models, the density immediately below the CMB is an interface value, not the volume-mean density of the whole unresolved core. A mantle reference-density table plus one CMB-side density does not determine total planetary mass.

The follow-up therefore requires an explicit positive total COM reference mass.

### 3. Internal potential and COM dipole still use different time layers

The COM dipole always traverses the current mechanical internal source. The full-domain potential suppresses internal sources on the initial update and enables them according to a separate iteration-number condition. This violates the stated one-source discrete contract.

The follow-up passes one `include_current_velocity_increment` flag through:

- volume-density source;
- displaced internal density interfaces;
- internal mass dipole;
- internal inertia tensor;
- radial Green moments.

Committed-only and committed-plus-current-predictor states are therefore evaluated consistently.

### 4. Convergence does not test the constraint residual

Current convergence checks coefficient changes only. The actual constraint residual

$$
\epsilon_D=
\frac{\|\mathbf D-M\mathbf c\|}
{\max(\|\mathbf D\|,100\epsilon MR)}
$$

must also pass. This is especially important with under-relaxation.

### 5. Fixed-inner models are rejected by an unrelated rotation policy

The code unconditionally requires `angular momentum` removal. That is valid for a fully free spherical shell with a rigid-rotation null mode, but not for a shell with a fixed inner boundary. Translational COM physics and toroidal rotational nullspace treatment are separate.

The follow-up continues to reject post-solve translation projectors but removes the unconditional rotational requirement.

### 6. Debug output has a thread race

The polar-wander assembler diagnostic writes from cell assembly using a shared static header flag. The follow-up serializes this debug-only file output with a mutex. A later cleanup should move the reduction and write completely outside threaded assembly.

## Required test order after applying the patch

1. Build `aspect-release`.
2. Run the existing default-none test.
3. Run native `Y10` COM with explicit mass.
4. Run the zero-mass expected-failure test.
5. Run the fixed-inner test without angular-momentum removal.
6. Run pure `Y20` and confirm COM remains at roundoff.
7. Repeat native `Y10` with relaxation factors 1.0 and 0.5 and verify the normalized residual decreases.
8. Repeat with MPI 1, 2, and 4 ranks.
9. Run PREM/VM5a `l1m0`, `l2m0`, and `l2m1` short gates.
10. Only after the short gates pass, resume the long-time `l2m0` leakage investigation.

## Merge condition

Do not treat a small algebraic `D-Mc` value as independent validation unless:

- the total mass contract is explicit and gravity-consistent;
- potential and dipole use the same source/time layer;
- the normalized residual participates in convergence;
- degree-two and higher kernels are unchanged when only the reference-frame option changes.
