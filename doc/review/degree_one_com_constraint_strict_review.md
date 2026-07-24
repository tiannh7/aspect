# Strict review: native coupled degree-one center-of-mass frame

Reviewed branch: `codex/degree-one-com-constraint`

Safe baseline commit: `4be6ab0d50a63eee512e4d16d075b9fee90e552f`

## Current decision

The native mode

```text
Potential feedback/Reference frame/Degree 1 reference frame = center of mass
```

must remain disabled until the contracts below are implemented and verified. The temporary initialization failure added in `4be6ab0d` is the correct short-term safety action.

The implementation direction is viable, but the previous acceptance test did not validate the physical COM constraint. It combined a unit surface gravity and radius 2 with order-one densities, so the normalization mass `g(R)R^2/G` exceeded the mass represented by the configured density model by about 1.6 billion. The resulting frame translation was consequently about 1.6 billion times too small.

## Blocking contracts

### 1. Explicit reference-mass contract

The constraint

$$
\mathbf D_h-M_{\mathrm{ref}}\mathbf c=0
$$

must use a total reference mass that is explicit and independent of arbitrary combinations of gravity, reference density, and CMB density parameters.

For a shell model that does not resolve the entire core, the total planetary mass generally cannot be reconstructed from the mantle reference-density table and the density immediately below the CMB. The native COM mode should therefore require an explicit positive parameter:

```text
Potential feedback/Reference frame/Center of mass reference mass
```

The configured surface-gravity mass

$$
M_g=\frac{g(R)R^2}{G}
$$

should remain a diagnostic consistency value, not the implicit normalization contract.

### 2. One source traversal and one time layer

For every potential iteration, all of the following must use the same source representation and the same time level:

- internal-density potential;
- internal mass dipole;
- internal inertia tensor;
- displaced internal density-interface sheets.

The old implementation included the internal dipole in the COM constraint while omitting the internal source from the first full-domain potential update. That violates the claimed coupled discrete system.

The source API should take an explicit `include_current_velocity_increment` flag. With the flag disabled, mechanical radial displacement must use committed history only. With it enabled, it must use committed history plus the current velocity predictor. The COM dipole, Green moments, and rotational moments must pass the same flag during the same nonlinear iteration.

### 3. Constraint residual convergence

Coefficient stagnation is not sufficient. Convergence must require all three conditions:

$$
\epsilon_\Phi \le \epsilon_{\Phi,\mathrm{tol}},
$$

$$
\epsilon_c \le \epsilon_{c,\mathrm{tol}},
$$

and

$$
\epsilon_D=
\frac{\|\mathbf D_h-M_{\mathrm{ref}}\mathbf c\|}
{\max(\|\mathbf D_h\|,100\epsilon M_{\mathrm{ref}}R)}
\le \epsilon_{D,\mathrm{tol}}.
$$

The residual must be printed independently of the coefficient-change diagnostic.

### 4. Rotational nullspace policy

The native COM mode must reject post-solve translational projections (`net translation` and `linear momentum`).

It must not unconditionally require `angular momentum` removal. A shell with a fixed inner boundary does not generally retain a rigid-rotation nullspace. Rotational nullspace treatment must follow the actual velocity boundary conditions and remain separate from the spheroidal degree-one COM constraint.

### 5. No threaded file output from cell assembly

The environment-controlled polar-wander RHS diagnostic currently writes directly from the Stokes cell assembler and uses a process-local static header flag. That is not thread safe. Cell-local values must be reduced through copy data or another thread-safe accumulator and written once outside the threaded assembly loop.

## Required verification matrix

The native mode must not be re-enabled until the following tests pass.

| Test | Required result |
|---|---|
| Analytic `Y10` load, self-consistent mass/gravity | `c_z=D_z/M_ref` with the expected sign and normalization |
| Analytic `Y11c` and `Y11s` loads | Correct Cartesian x/y mapping |
| Pure `Y20` load | COM coefficients remain at roundoff and the target Love numbers are unchanged |
| Internal pure `Y10` source | Dipole, full-domain potential, and COM coefficient use the same discrete source |
| Surface + CMB + internal source cancellation | Independent source contributions sum to the expected zero/nonzero dipole |
| Relaxation factor below 1 | Constraint residual decreases and is part of convergence |
| Free inner and outer boundaries | Appropriate rotational nullspace treatment succeeds |
| Fixed inner, free outer boundary | Native COM works without forced angular-momentum projection |
| Quadrature point / cell average / mass-lumped radial layer | Potential and low-order moments use identical source discretization |
| MPI 1/2/4 ranks | COM coefficients and residual are partition invariant |
| Restart | COM state and convergence are reproducible |
| PREM/VM5a `l1m0`, `l2m0`, `l2m1` short gates | Existing benchmark accuracy is preserved |
| PREM/VM5a `l2m0` long run | Native COM does not introduce or hide high-degree growth |

## Merge policy

1. Keep `center of mass` disabled on this branch until the implementation patch is applied and the focused tests pass.
2. Do not restore the old success baseline; its normalization was physically inconsistent.
3. Add an expected-failure regression for the disabled mode, or restore valid success tests only together with the complete implementation.
4. Do not use high-degree filtering, post-solve translation removal, or diagnostic subtraction as a substitute for the coupled constraint.
5. Do not claim that a small algebraic `D-Mc` residual validates the mass contract when `M` is derived from a different physical model.

A draft implementation patch is stored alongside this review as `degree_one_com_constraint_review_fixes.patch`. It is intentionally not applied automatically because the repository must be built and the generated ASPECT test baselines must be regenerated in a configured local worktree.