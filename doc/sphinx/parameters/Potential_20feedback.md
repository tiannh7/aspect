(parameters:Potential_20feedback)=
# Potential feedback


## **Subsection:** Potential feedback


::::{dropdown} __Parameter:__ {ref}`List of feedback mechanisms<parameters:Potential_20feedback/List_20of_20feedback_20mechanisms>`
:name: parameters:Potential_20feedback/List_20of_20feedback_20mechanisms
**Default value:**

**Pattern:** [List of <[Selection self gravity|tidal potential|rotational feedback|glacial isostatic adjustment ]> of length 0...4294967295 (inclusive)]

**Documentation:** Comma-separated list of active potential-feedback mechanisms. Supported names are &lsquo;self gravity&rsquo;, &lsquo;tidal potential&rsquo;, &lsquo;rotational feedback&rsquo;, and &lsquo;glacial isostatic adjustment&rsquo;. Mechanisms are activated by this list, not by per-mechanism Enable flags.
::::

(parameters:Potential_20feedback/Glacial_20isostatic_20adjustment)=
## **Subsection:** Potential feedback / Glacial isostatic adjustment
::::{dropdown} __Parameter:__ {ref}`Ice density<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20density>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20density
**Default value:** 917.4

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Ice density in kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Ice load reference<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20load_20reference>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20load_20reference
**Default value:** first history file

**Pattern:** [Selection first history file|zero thickness ]

**Documentation:** Reference ice load. &lsquo;first history file&rsquo; applies changes relative to the first stage; &lsquo;zero thickness&rsquo; applies the absolute ice history.
::::

::::{dropdown} __Parameter:__ {ref}`Maximum degree<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Maximum_20degree>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Maximum_20degree
**Default value:** 32

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Maximum spherical-harmonic degree retained for ice, ocean, and total GIA surface loads.
::::

::::{dropdown} __Parameter:__ {ref}`Water density<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Water_20density>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Water_20density
**Default value:** 1000.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Ocean-water density in kg/m^3.
::::

(parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history)=
## **Subsection:** Potential feedback / Glacial isostatic adjustment / Ice history
::::{dropdown} __Parameter:__ {ref}`Data directory<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Data_20directory>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Data_20directory
**Default value:** $ASPECT_SOURCE_DIR/data/potential-feedback/gia/

**Pattern:** [DirectoryName]

**Documentation:** Directory containing the structured surface history files.
::::

::::{dropdown} __Parameter:__ {ref}`Data file name<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Data_20file_20name>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Data_20file_20name
**Default value:** ice.%d.txt

**Pattern:** [Anything]

**Documentation:** File name or printf-style integer pattern for the structured surface fields.
::::

::::{dropdown} __Parameter:__ {ref}`Data format<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Data_20format>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Data_20format
**Default value:** aspect structured data

**Pattern:** [Selection aspect structured data|citcomsve regular grid ]

**Documentation:** Input format. &lsquo;aspect structured data&rsquo; uses the standard ASPECT # POINTS format with coordinates in radians. &lsquo;citcomsve regular grid&rsquo; reads the canonical CitcomSVE nlon-by-nlat longitude/latitude grid in degrees.
::::

::::{dropdown} __Parameter:__ {ref}`First data file number<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/First_20data_20file_20number>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/First_20data_20file_20number
**Default value:** 0

**Pattern:** [Integer range -2147483648...2147483647 (inclusive)]

**Documentation:** File number of the first CitcomSVE stage or of a single static field whose name contains an integer placeholder.
::::

::::{dropdown} __Parameter:__ {ref}`Interpolate between stages<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Interpolate_20between_20stages>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Interpolate_20between_20stages
**Default value:** true

**Pattern:** [Bool]

**Documentation:** Whether to linearly interpolate fields between successive schedule stages.
::::

::::{dropdown} __Parameter:__ {ref}`Scale factor<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Scale_20factor>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Scale_20factor
**Default value:** 1.0

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Multiplicative scale applied to values read from the structured data files.
::::

::::{dropdown} __Parameter:__ {ref}`Schedule file name<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Schedule_20file_20name>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Schedule_20file_20name
**Default value:**

**Pattern:** [Anything]

**Documentation:** Schedule file. An empty value selects a single static field.
::::

::::{dropdown} __Parameter:__ {ref}`Schedule format<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Schedule_20format>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Ice_20history/Schedule_20format
**Default value:** elapsed time and file number

**Pattern:** [Selection elapsed time and file number|citcomsve stage ages ]

**Documentation:** Schedule syntax. Elapsed-time schedules contain time and file number columns. CitcomSVE schedules contain a stage-interval-count header followed by one more row of age in ka and stage time-step count; files are numbered sequentially.
::::

(parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history)=
## **Subsection:** Potential feedback / Glacial isostatic adjustment / Prescribed ocean function history
::::{dropdown} __Parameter:__ {ref}`Data directory<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Data_20directory>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Data_20directory
**Default value:** $ASPECT_SOURCE_DIR/data/potential-feedback/gia/

**Pattern:** [DirectoryName]

**Documentation:** Directory containing the structured surface history files.
::::

::::{dropdown} __Parameter:__ {ref}`Data file name<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Data_20file_20name>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Data_20file_20name
**Default value:** ocean.%d.txt

**Pattern:** [Anything]

**Documentation:** File name or printf-style integer pattern for the structured surface fields.
::::

::::{dropdown} __Parameter:__ {ref}`Data format<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Data_20format>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Data_20format
**Default value:** aspect structured data

**Pattern:** [Selection aspect structured data|citcomsve regular grid ]

**Documentation:** Input format. &lsquo;aspect structured data&rsquo; uses the standard ASPECT # POINTS format with coordinates in radians. &lsquo;citcomsve regular grid&rsquo; reads the canonical CitcomSVE nlon-by-nlat longitude/latitude grid in degrees.
::::

::::{dropdown} __Parameter:__ {ref}`First data file number<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/First_20data_20file_20number>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/First_20data_20file_20number
**Default value:** 0

**Pattern:** [Integer range -2147483648...2147483647 (inclusive)]

**Documentation:** File number of the first CitcomSVE stage or of a single static field whose name contains an integer placeholder.
::::

::::{dropdown} __Parameter:__ {ref}`Interpolate between stages<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Interpolate_20between_20stages>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Interpolate_20between_20stages
**Default value:** true

**Pattern:** [Bool]

**Documentation:** Whether to linearly interpolate fields between successive schedule stages.
::::

::::{dropdown} __Parameter:__ {ref}`Scale factor<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Scale_20factor>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Scale_20factor
**Default value:** 1.0

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Multiplicative scale applied to values read from the structured data files.
::::

::::{dropdown} __Parameter:__ {ref}`Schedule file name<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Schedule_20file_20name>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Schedule_20file_20name
**Default value:**

**Pattern:** [Anything]

**Documentation:** Schedule file. An empty value selects a single static field.
::::

::::{dropdown} __Parameter:__ {ref}`Schedule format<parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Schedule_20format>`
:name: parameters:Potential_20feedback/Glacial_20isostatic_20adjustment/Prescribed_20ocean_20function_20history/Schedule_20format
**Default value:** elapsed time and file number

**Pattern:** [Selection elapsed time and file number|citcomsve stage ages ]

**Documentation:** Schedule syntax. Elapsed-time schedules contain time and file number columns. CitcomSVE schedules contain a stage-interval-count header followed by one more row of age in ka and stage time-step count; files are numbered sequentially.
::::

(parameters:Potential_20feedback/Potential_20iteration)=
## **Subsection:** Potential feedback / Potential iteration
::::{dropdown} __Parameter:__ {ref}`Freeze after timestep zero<parameters:Potential_20feedback/Potential_20iteration/Freeze_20after_20timestep_20zero>`
:name: parameters:Potential_20feedback/Potential_20iteration/Freeze_20after_20timestep_20zero
**Alias:** [Freeze feedback after timestep zero](parameters:Potential_20feedback/Potential_20iteration/Freeze_20feedback_20after_20timestep_20zero)

**Deprecation Status:** false
::::

::::{dropdown} __Parameter:__ {ref}`Freeze feedback after timestep zero<parameters:Potential_20feedback/Potential_20iteration/Freeze_20feedback_20after_20timestep_20zero>`
:name: parameters:Potential_20feedback/Potential_20iteration/Freeze_20feedback_20after_20timestep_20zero
**Default value:** false

**Pattern:** [Bool]

**Documentation:** If true, retain the converged timestep-zero feedback potential at later timesteps.
::::

::::{dropdown} __Parameter:__ {ref}`Initial displacement time step<parameters:Potential_20feedback/Potential_20iteration/Initial_20displacement_20time_20step>`
:name: parameters:Potential_20feedback/Potential_20iteration/Initial_20displacement_20time_20step
**Default value:** 0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Displacement interval used to convert the timestep-0 Stokes velocity into an incremental boundary displacement in seconds. If zero, feedback mechanisms use the material model&rsquo;s initial elastic time step when available.
::::

::::{dropdown} __Parameter:__ {ref}`Iterate with Stokes<parameters:Potential_20feedback/Potential_20iteration/Iterate_20with_20Stokes>`
:name: parameters:Potential_20feedback/Potential_20iteration/Iterate_20with_20Stokes
**Default value:** true

**Pattern:** [Bool]

**Documentation:** Recompute feedback potentials from the current Stokes velocity after every Stokes solve.
::::

::::{dropdown} __Parameter:__ {ref}`Maximum iterations<parameters:Potential_20feedback/Potential_20iteration/Maximum_20iterations>`
:name: parameters:Potential_20feedback/Potential_20iteration/Maximum_20iterations
**Default value:** 20

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Maximum number of self-consistent potential updates per timestep. The iteration stops when all active feedback-potential coefficient vectors reach the relative tolerance or this limit is reached.
::::

::::{dropdown} __Parameter:__ {ref}`Relative tolerance<parameters:Potential_20feedback/Potential_20iteration/Relative_20tolerance>`
:name: parameters:Potential_20feedback/Potential_20iteration/Relative_20tolerance
**Default value:** 1e-3

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Relative change tolerance for mechanism-specific feedback-potential coefficient vectors.
::::

::::{dropdown} __Parameter:__ {ref}`Relaxation factor<parameters:Potential_20feedback/Potential_20iteration/Relaxation_20factor>`
:name: parameters:Potential_20feedback/Potential_20iteration/Relaxation_20factor
**Default value:** 1.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Under-relaxation factor for potential iteration coefficient updates.
::::

(parameters:Potential_20feedback/Reference_20frame)=
## **Subsection:** Potential feedback / Reference frame
::::{dropdown} __Parameter:__ {ref}`Degree 1 reference frame<parameters:Potential_20feedback/Reference_20frame/Degree_201_20reference_20frame>`
:name: parameters:Potential_20feedback/Reference_20frame/Degree_201_20reference_20frame
**Default value:** none

**Pattern:** [Selection none|geoid cancellation|center of mass|citcomsve center of mass ]

**Documentation:** Degree-1 potential/reference-frame convention. &lsquo;none&rsquo; leaves degree-1 potential unmodified. &lsquo;geoid cancellation&rsquo; removes degree-1 potential from the emitted boundary traction potential. &lsquo;center of mass&rsquo; applies an ASPECT-native reference-frame correction from the total degree-1 mass dipole of active self-gravity mass sources. &lsquo;citcomsve center of mass&rsquo; keeps the benchmark-compatible CitcomSVE incompressible degree-1 load-compensation replay.
::::

::::{dropdown} __Parameter:__ {ref}`Remove pure rotation from displacement<parameters:Potential_20feedback/Reference_20frame/Remove_20pure_20rotation_20from_20displacement>`
:name: parameters:Potential_20feedback/Reference_20frame/Remove_20pure_20rotation_20from_20displacement
**Default value:** true

**Pattern:** [Bool]

**Documentation:** Whether to remove pure-rotation reference-frame content from displacement diagnostics. This is separate from ASPECT velocity nullspace removal.
::::

(parameters:Potential_20feedback/Rotational_20feedback)=
## **Subsection:** Potential feedback / Rotational feedback
::::{dropdown} __Parameter:__ {ref}`Fluid Love number<parameters:Potential_20feedback/Rotational_20feedback/Fluid_20Love_20number>`
:name: parameters:Potential_20feedback/Rotational_20feedback/Fluid_20Love_20number
**Default value:** 1.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Fluid degree-2 Love number k_f used in the linearized polar-wander relation. This is the same quantity as CitcomSVE&rsquo;s polar_wander_kf. Rotational feedback is internally the degree-2, order-1 polar-wander forcing used by CitcomSVE.
::::

(parameters:Potential_20feedback/Self_20gravity)=
## **Subsection:** Potential feedback / Self gravity
::::{dropdown} __Parameter:__ {ref}`Density above CMB<parameters:Potential_20feedback/Self_20gravity/Density_20above_20CMB>`
:name: parameters:Potential_20feedback/Self_20gravity/Density_20above_20CMB
**Default value:** 4604.4

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** density immediately above the CMB in kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Density above surface<parameters:Potential_20feedback/Self_20gravity/Density_20above_20surface>`
:name: parameters:Potential_20feedback/Self_20gravity/Density_20above_20surface
**Default value:** 0.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** density immediately above the surface in kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Density below CMB<parameters:Potential_20feedback/Self_20gravity/Density_20below_20CMB>`
:name: parameters:Potential_20feedback/Self_20gravity/Density_20below_20CMB
**Default value:** 10005.4

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** density immediately below the CMB in kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Density below surface<parameters:Potential_20feedback/Self_20gravity/Density_20below_20surface>`
:name: parameters:Potential_20feedback/Self_20gravity/Density_20below_20surface
**Default value:** 4604.4

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** density immediately below the surface in kg/m^3.
::::

::::{dropdown} __Parameter:__ {ref}`Full domain volume source discretization<parameters:Potential_20feedback/Self_20gravity/Full_20domain_20volume_20source_20discretization>`
:name: parameters:Potential_20feedback/Self_20gravity/Full_20domain_20volume_20source_20discretization
**Default value:** quadrature point

**Pattern:** [Selection quadrature point|cell average|radial layer midpoint|mass lumped radial layer ]

**Documentation:** Discretization of the mechanical volume-density source in the 3-D full-domain self-gravity potential. &lsquo;quadrature point&rsquo; preserves the existing pointwise integration. &lsquo;cell average&rsquo; uses one volume-weighted density perturbation per active cell before applying the spherical-harmonic Green kernel. &lsquo;radial layer midpoint&rsquo; additionally uses an arithmetic quadrature-point source average and evaluates the radial kernel and radial measure at the cell&rsquo;s midpoint radius. &lsquo;mass lumped radial layer&rsquo; first projects those cell averages to shared pressure vertices with a lumped Q1 mass matrix before applying the midpoint rule. The default is unchanged.
::::

::::{dropdown} __Parameter:__ {ref}`Include internal density anomalies<parameters:Potential_20feedback/Self_20gravity/Include_20internal_20density_20anomalies>`
:name: parameters:Potential_20feedback/Self_20gravity/Include_20internal_20density_20anomalies
**Default value:** auto

**Pattern:** [Selection true|false|auto ]

**Documentation:** controls whether internal volume density anomalies contribute to self-gravity feedback and geoid diagnostics. With 3-D mechanical mass conservation, the resulting potential is also applied through the full-domain compressible weak form.
::::

::::{dropdown} __Parameter:__ {ref}`Internal density anomaly tolerance<parameters:Potential_20feedback/Self_20gravity/Internal_20density_20anomaly_20tolerance>`
:name: parameters:Potential_20feedback/Self_20gravity/Internal_20density_20anomaly_20tolerance
**Default value:** 0.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** absolute tolerance for auto-detecting zero internal density anomaly field.
::::

::::{dropdown} __Parameter:__ {ref}`Maximum degree<parameters:Potential_20feedback/Self_20gravity/Maximum_20degree>`
:name: parameters:Potential_20feedback/Self_20gravity/Maximum_20degree
**Default value:** 32

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Maximum spherical harmonic degree retained for self-gravity potential feedback. The calculation starts internally at degree 1, matching CitcomSVE&rsquo;s self-gravity potential synthesis.
::::

::::{dropdown} __Parameter:__ {ref}`Reference density for internal anomalies<parameters:Potential_20feedback/Self_20gravity/Reference_20density_20for_20internal_20anomalies>`
:name: parameters:Potential_20feedback/Self_20gravity/Reference_20density_20for_20internal_20anomalies
**Default value:** 0.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** A constant spherically symmetric reference density subtracted from the volume-density integral.
::::

(parameters:Potential_20feedback/Self_2dconsistent_20potential_20update)=
## **Subsection:** Potential feedback / Self-consistent potential update
::::{dropdown} __Parameter:__ {ref}`Freeze feedback after timestep zero<parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Freeze_20feedback_20after_20timestep_20zero>`
:name: parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Freeze_20feedback_20after_20timestep_20zero
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Deprecated compatibility alias for Potential iteration/Freeze after timestep zero.
::::

::::{dropdown} __Parameter:__ {ref}`Iterate with Stokes<parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Iterate_20with_20Stokes>`
:name: parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Iterate_20with_20Stokes
**Default value:** true

**Pattern:** [Bool]

**Documentation:** Deprecated compatibility alias for Potential iteration/Iterate with Stokes.
::::

::::{dropdown} __Parameter:__ {ref}`Maximum iterations<parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Maximum_20iterations>`
:name: parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Maximum_20iterations
**Default value:** 20

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Deprecated compatibility alias for Potential iteration/Maximum iterations.
::::

::::{dropdown} __Parameter:__ {ref}`Relative tolerance<parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Relative_20tolerance>`
:name: parameters:Potential_20feedback/Self_2dconsistent_20potential_20update/Relative_20tolerance
**Default value:** 1e-3

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Deprecated compatibility alias for Potential iteration/Relative tolerance.
::::

(parameters:Potential_20feedback/Tidal_20potential)=
## **Subsection:** Potential feedback / Tidal potential
::::{dropdown} __Parameter:__ {ref}`Model name<parameters:Potential_20feedback/Tidal_20potential/Model_20name>`
:name: parameters:Potential_20feedback/Tidal_20potential/Model_20name
**Default value:** none

**Pattern:** [Selection none|spherical harmonic potential ]

**Documentation:** Select the tidal-potential model. The &lsquo;none&rsquo; model disables externally prescribed tidal potential. The &lsquo;spherical harmonic potential&rsquo; model adds one real spherical-harmonic Phi/g coefficient directly to the potential-feedback coefficient arrays.
::::

(parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential)=
## **Subsection:** Potential feedback / Tidal potential / Spherical harmonic potential
::::{dropdown} __Parameter:__ {ref}`Angular frequency<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Angular_20frequency>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Angular_20frequency
**Default value:** 0.0

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Angular frequency for sinusoidal time dependence.
::::

::::{dropdown} __Parameter:__ {ref}`Coefficient type<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Coefficient_20type>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Coefficient_20type
**Default value:** cosine

**Pattern:** [Selection cosine|sine ]

**Documentation:** Select the real spherical harmonic coefficient type for the tidal potential.
::::

::::{dropdown} __Parameter:__ {ref}`Harmonic degree<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Harmonic_20degree>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Harmonic_20degree
**Default value:** 2

**Pattern:** [Integer range 0...2147483647 (inclusive)]

**Documentation:** Spherical harmonic degree of the tidal potential.
::::

::::{dropdown} __Parameter:__ {ref}`Harmonic order<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Harmonic_20order>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Harmonic_20order
**Default value:** 0

**Pattern:** [Integer range 0...2147483647 (inclusive)]

**Documentation:** Spherical harmonic order of the tidal potential.
::::

::::{dropdown} __Parameter:__ {ref}`Normalization<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Normalization>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Normalization
**Default value:** geodesy 4pi

**Pattern:** [Selection geodesy 4pi|unnormalized legendre ]

**Documentation:** Normalization of the tidal potential. The &lsquo;geodesy 4pi&rsquo; option uses ASPECT&rsquo;s Utilities::real_spherical_harmonic convention. The &lsquo;unnormalized legendre&rsquo; option prescribes P_l(cos theta) and is restricted to m=0.
::::

::::{dropdown} __Parameter:__ {ref}`Phase<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Phase>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Phase
**Default value:** 0.0

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Phase for sinusoidal time dependence in radians.
::::

::::{dropdown} __Parameter:__ {ref}`Potential amplitude<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Potential_20amplitude>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Potential_20amplitude
**Default value:** 0.0

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Amplitude of Phi at the outer surface in m^2/s^2. Used when Potential quantity is &lsquo;potential&rsquo;.
::::

::::{dropdown} __Parameter:__ {ref}`Potential height amplitude<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Potential_20height_20amplitude>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Potential_20height_20amplitude
**Default value:** 0.0

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Amplitude of Phi/g at the outer surface in meters.
::::

::::{dropdown} __Parameter:__ {ref}`Potential quantity<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Potential_20quantity>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Potential_20quantity
**Default value:** potential height

**Pattern:** [Selection potential height|potential ]

**Documentation:** Whether the input amplitude is Phi/g in meters or Phi in m^2/s^2.
::::

::::{dropdown} __Parameter:__ {ref}`Reference gravity<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Reference_20gravity>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Reference_20gravity
**Default value:** 1.0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Reference gravity used to convert Phi to Phi/g.
::::

::::{dropdown} __Parameter:__ {ref}`Time dependence<parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Time_20dependence>`
:name: parameters:Potential_20feedback/Tidal_20potential/Spherical_20harmonic_20potential/Time_20dependence
**Default value:** none

**Pattern:** [Selection none|sinusoidal ]

**Documentation:** Whether the tidal potential is static or multiplied by cos(Angular frequency * time + Phase).
::::
