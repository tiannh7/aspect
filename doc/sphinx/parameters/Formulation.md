(parameters:Formulation)=
# Formulation


## **Subsection:** Formulation


::::{dropdown} __Parameter:__ {ref}`Enable additional Stokes RHS<parameters:Formulation/Enable_20additional_20Stokes_20RHS>`
:name: parameters:Formulation/Enable_20additional_20Stokes_20RHS
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Whether to ask the material model for additional terms for the right-hand side of the Stokes equation. This feature is likely only used when implementing force vectors for manufactured solution problems and requires filling additional outputs of type AdditionalMaterialOutputsStokesRHS.
::::

::::{dropdown} __Parameter:__ {ref}`Enable elasticity<parameters:Formulation/Enable_20elasticity>`
:name: parameters:Formulation/Enable_20elasticity
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Whether to include the additional elastic terms on the right-hand side of the Stokes equation.
::::

::::{dropdown} __Parameter:__ {ref}`Enable prescribed dilation<parameters:Formulation/Enable_20prescribed_20dilation>`
:name: parameters:Formulation/Enable_20prescribed_20dilation
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Whether to include additional terms on the right-hand side of the Stokes equation to set a given compression term specified in the MaterialModel output PrescribedPlasticDilation.
::::

::::{dropdown} __Parameter:__ {ref}`Formulation<parameters:Formulation/Formulation>`
:name: parameters:Formulation/Formulation
**Default value:** custom

**Pattern:** [Selection isentropic compression|custom|anelastic liquid approximation|Boussinesq approximation ]

**Documentation:** Select a formulation for the basic equations. Different published formulations are available in ASPECT (see the list of possible values for this parameter in the manual for available options). Two ASPECT specific options are
  * &lsquo;isentropic compression&rsquo;: ASPECT&rsquo;s original formulation, using the explicit compressible mass equation, and the full density for the temperature equation.
  * &lsquo;custom&rsquo;: A custom selection of &lsquo;Mass conservation&rsquo; and &lsquo;Temperature equation&rsquo;.
:::{warning}
The &lsquo;custom&rsquo; option is implemented for advanced users that want full control over the equations solved. It is possible to choose inconsistent formulations and no error checking is performed on the consistency of the resulting equations.
:::

:::{note}
The &lsquo;anelastic liquid approximation&rsquo; option here can also be used to set up the &lsquo;truncated anelastic liquid approximation&rsquo; as long as this option is chosen together with a material model that defines a density that depends on temperature and depth and not on the pressure.
:::
::::

::::{dropdown} __Parameter:__ {ref}`Mass conservation<parameters:Formulation/Mass_20conservation>`
:name: parameters:Formulation/Mass_20conservation
**Default value:** ask material model

**Pattern:** [Selection incompressible|isentropic compression|hydrostatic compression|reference density profile|implicit reference density profile|projected density field|elastic pressure evolution|ask material model ]

**Documentation:** Possible approximations for the density derivatives in the mass conservation equation. &lsquo;Elastic pressure evolution&rsquo; implements dp&rsquo;/dt=-K div(v) using the Stokes pressure unknown and an elastic bulk modulus supplied by the material model; it is distinct from thermodynamic &lsquo;isentropic compression&rsquo;. Note that this parameter is only evaluated if &lsquo;Formulation&rsquo; is set to &lsquo;custom&rsquo;. Other formulations ignore the value of this parameter.
::::

::::{dropdown} __Parameter:__ {ref}`Temperature equation<parameters:Formulation/Temperature_20equation>`
:name: parameters:Formulation/Temperature_20equation
**Default value:** real density

**Pattern:** [Selection real density|reference density profile ]

**Documentation:** Possible approximations for the density in the temperature equation. Possible approximations are &lsquo;real density&rsquo; and &lsquo;reference density profile&rsquo;. Note that this parameter is only evaluated if &lsquo;Formulation&rsquo; is set to &lsquo;custom&rsquo;. Other formulations ignore the value of this parameter.
::::

(parameters:Formulation/Density_20sources)=
## **Subsection:** Formulation / Density sources
::::{dropdown} __Parameter:__ {ref}`Allow viscoelastic initial mechanical response<parameters:Formulation/Density_20sources/Allow_20viscoelastic_20initial_20mechanical_20response>`
:name: parameters:Formulation/Density_20sources/Allow_20viscoelastic_20initial_20mechanical_20response
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Whether mechanical mass conservation may use the material model&rsquo;s finite viscoelastic response during the timestep-zero loaded solve. The default false requires the instantaneous elastic initial response. Enable this only for an explicit time-integration discriminator; it does not change later viscoelastic timesteps.
::::

::::{dropdown} __Parameter:__ {ref}`Constant reference density<parameters:Formulation/Density_20sources/Constant_20reference_20density>`
:name: parameters:Formulation/Density_20sources/Constant_20reference_20density
**Default value:** 0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Constant central reference density used when &lsquo;Reference density model&rsquo; is &lsquo;constant&rsquo;. Units: kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Density source law<parameters:Formulation/Density_20sources/Density_20source_20law>`
:name: parameters:Formulation/Density_20sources/Density_20source_20law
**Default value:** legacy

**Pattern:** [Selection legacy|material density|material minus reference|zero volume perturbation|mechanical mass conservation ]

**Documentation:** Select the volume-density source consumed by Stokes momentum, internal self-gravity, and centre-of-mass/degree-1 integrals. &lsquo;legacy&rsquo; preserves each consumer&rsquo;s historical behavior. &lsquo;material density&rsquo; uses physical material density, &lsquo;material minus reference&rsquo; subtracts the selected central reference, and &lsquo;zero volume perturbation&rsquo; suppresses volume-density sources without changing explicit surface or CMB sheet and restoring terms. &lsquo;mechanical mass conservation&rsquo; reconstructs the density perturbation from elastic pressure and radial material-displacement history. It is intended for radial reference states and requires the &lsquo;elastic pressure evolution&rsquo; mass formulation.
::::

::::{dropdown} __Parameter:__ {ref}`Frozen reference density profile slices<parameters:Formulation/Density_20sources/Frozen_20reference_20density_20profile_20slices>`
:name: parameters:Formulation/Density_20sources/Frozen_20reference_20density_20profile_20slices
**Default value:** 100

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Number of finite geometric-depth bins in the frozen initial lateral-average reference-density profile. Values are stored at bin centers and linearly interpolated with endpoint clamping. Units: none.
::::

::::{dropdown} __Parameter:__ {ref}`Internal density jump density contrasts<parameters:Formulation/Density_20sources/Internal_20density_20jump_20density_20contrasts>`
:name: parameters:Formulation/Density_20sources/Internal_20density_20jump_20density_20contrasts
**Default value:**

**Pattern:** [List of <[Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]> of length 0...4294967295 (inclusive)]

**Documentation:** Density below minus density above each radius in &lsquo;Internal density jump radii&rsquo;. Negative contrasts are allowed. The two lists must have equal length. Units: kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Internal density jump face tolerance<parameters:Formulation/Density_20sources/Internal_20density_20jump_20face_20tolerance>`
:name: parameters:Formulation/Density_20sources/Internal_20density_20jump_20face_20tolerance
**Default value:** 1

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Absolute radial tolerance used to match an explicitly configured internal density jump to a mesh face. Piecewise-constant tabulated jumps instead use adjacent cell identities and remain active under mesh deformation. This value must be positive. Units: m.
::::

::::{dropdown} __Parameter:__ {ref}`Internal density jump radii<parameters:Formulation/Density_20sources/Internal_20density_20jump_20radii>`
:name: parameters:Formulation/Density_20sources/Internal_20density_20jump_20radii
**Default value:**

**Pattern:** [List of <[Double 0...MAX_DOUBLE (inclusive)]> of length 0...4294967295 (inclusive)]

**Documentation:** Strictly increasing radii of sharp internal density interfaces. The default empty list disables explicitly prescribed interface sheets; piecewise-constant tabulated reference density still derives its internal sheets automatically. Each configured radius must coincide with a radial mesh face. Units: m.
::::

::::{dropdown} __Parameter:__ {ref}`Reference density model<parameters:Formulation/Density_20sources/Reference_20density_20model>`
:name: parameters:Formulation/Density_20sources/Reference_20density_20model
**Default value:** none

**Pattern:** [Selection none|constant|frozen initial lateral average|tabulated radial ]

**Documentation:** Select the central reference-density model used by non-legacy volume-density source laws. &lsquo;none&rsquo; defines a zero reference, &lsquo;constant&rsquo; uses the configured constant value, and &lsquo;frozen initial lateral average&rsquo; computes one quadrature-weighted initial lateral average in geometric-depth bins and freezes it for the run. &lsquo;tabulated radial&rsquo; uses the configured radius and density lists with the selected interpolation.
::::

::::{dropdown} __Parameter:__ {ref}`Tabulated mechanical gravity magnitudes<parameters:Formulation/Density_20sources/Tabulated_20mechanical_20gravity_20magnitudes>`
:name: parameters:Formulation/Density_20sources/Tabulated_20mechanical_20gravity_20magnitudes
**Default value:**

**Pattern:** [List of <[Double 0...MAX_DOUBLE (inclusive)]> of length 0...4294967295 (inclusive)]

**Documentation:** Optional gravity magnitudes for mechanical mass conservation, with one constant value for each interval in &lsquo;Tabulated reference radii&rsquo;. The default empty list uses the selected gravity model at volume quadrature points. A nonempty list changes only the local mechanical volume couplings; boundary and internal-interface restoring terms continue to use the selected gravity model. Units: m/s^2.
::::

::::{dropdown} __Parameter:__ {ref}`Tabulated reference densities<parameters:Formulation/Density_20sources/Tabulated_20reference_20densities>`
:name: parameters:Formulation/Density_20sources/Tabulated_20reference_20densities
**Default value:** 0, 0

**Pattern:** [List of <[Double 0...MAX_DOUBLE (inclusive)]> of length 0...4294967295 (inclusive)]

**Documentation:** Reference densities corresponding to &lsquo;Tabulated reference radii&rsquo;. Units: kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Tabulated reference density interpolation<parameters:Formulation/Density_20sources/Tabulated_20reference_20density_20interpolation>`
:name: parameters:Formulation/Density_20sources/Tabulated_20reference_20density_20interpolation
**Default value:** linear

**Pattern:** [Selection linear|piecewise constant ]

**Documentation:** Interpolation used between tabulated reference radii. &lsquo;Linear&rsquo; preserves the continuous piecewise-linear reference profile. &lsquo;Piecewise constant&rsquo; uses the density at the lower radius over each outward interval and automatically treats every interior table radius with unequal adjacent densities as a sharp density sheet. The default is &lsquo;linear&rsquo;.
::::

::::{dropdown} __Parameter:__ {ref}`Tabulated reference radii<parameters:Formulation/Density_20sources/Tabulated_20reference_20radii>`
:name: parameters:Formulation/Density_20sources/Tabulated_20reference_20radii
**Default value:** 0, 1

**Pattern:** [List of <[Double 0...MAX_DOUBLE (inclusive)]> of length 0...4294967295 (inclusive)]

**Documentation:** Strictly increasing radii for the tabulated radial reference state. At least two values are required. Units: m.
::::

(parameters:Formulation/Elasticity)=
## **Subsection:** Formulation / Elasticity
::::{dropdown} __Parameter:__ {ref}`Use old stress fields<parameters:Formulation/Elasticity/Use_20old_20stress_20fields>`
:name: parameters:Formulation/Elasticity/Use_20old_20stress_20fields
**Default value:** true

**Pattern:** [Bool]

**Documentation:** Whether to use the old stress fields for interpolation when computational and elastic timesteps differ. If false, assumes computational timestep equals elastic timestep, simplifying the model by not using the old stress fields. This parameter is only relevant when using viscoelastic material models.
::::

(parameters:Formulation/Stokes_20pressure)=
## **Subsection:** Formulation / Stokes pressure
::::{dropdown} __Parameter:__ {ref}`Pressure formulation<parameters:Formulation/Stokes_20pressure/Pressure_20formulation>`
:name: parameters:Formulation/Stokes_20pressure/Pressure_20formulation
**Default value:** total pressure

**Pattern:** [Selection total pressure|dynamic pressure ]

**Documentation:** Select whether the Stokes pressure/body-force formulation uses total pressure or dynamic pressure. In total-pressure mode, the full material density contributes to the body force rho*g. In dynamic-pressure mode, a constant reference density is subtracted so that the Stokes body force is (rho - rho_ref)*g. Dynamic pressure mode requires <Formulation/Enable additional Stokes RHS = true>.
::::

::::{dropdown} __Parameter:__ {ref}`Reference density<parameters:Formulation/Stokes_20pressure/Reference_20density>`
:name: parameters:Formulation/Stokes_20pressure/Reference_20density
**Default value:** 0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Reference density used only when Pressure formulation is dynamic pressure. This value is subtracted from the material density in the Stokes body force through the additional Stokes RHS. Units: kg/m^3.
::::
