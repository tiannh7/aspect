/*
  Copyright (C) 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_potential_feedback_self_gravitation_h
#define _aspect_potential_feedback_self_gravitation_h

#include <aspect/potential_feedback/tidal_potential.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/potential_feedback/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace internal
    {
      /**
       * Accumulate the radial Green-function moments of spherical-harmonic
       * source coefficients. The two moment arrays permit evaluating all
       * configured radii with prefix and suffix sums instead of applying the
       * Green kernel separately at every source/evaluation-radius pair.
       */
      class RadialGreenMomentAccumulator
      {
        public:
          RadialGreenMomentAccumulator(
            const std::vector<double> &evaluation_radii,
            const unsigned int minimum_degree,
            const unsigned int maximum_degree,
            const double reference_radius);

          void
          add_source(const double source_radius,
                     const std::vector<double> &source_cos_coefficients,
                     const std::vector<double> &source_sin_coefficients);

          void
          mpi_sum(const MPI_Comm &mpi_communicator);

          std::pair<std::vector<double>, std::vector<double>>
          evaluate() const;

        private:
          std::vector<double> evaluation_radii;
          std::vector<unsigned int> coefficient_degrees;
          double reference_radius;
          unsigned int n_coefficients;
          std::vector<double> inner_cos_moments;
          std::vector<double> inner_sin_moments;
          std::vector<double> outer_cos_moments;
          std::vector<double> outer_sin_moments;
      };
    }



    /**
     * A boundary traction plugin that computes the self-gravitational
     * feedback from surface topography. Supports both 3D (spherical shell,
     * using spherical harmonic expansion) and 2D (annulus, using Fourier
     * expansion).
     *
     * The density contrasts at the surface and CMB are:
     *   delta_rho_surf = density_below_surface - density_above_surface
     *   delta_rho_cmb  = density_below_CMB - density_above_CMB
     *
     * For example (surface):
     *   - Rock topography in vacuum: density_below_surface=3500, density_above_surface=0
     *   - Ice cap on rock: density_below_surface=917, density_above_surface=0
     * For example (CMB):
     *   - Earth CMB: density_above_CMB=5500 (lower mantle), density_below_CMB=9900 (outer core)
     *   - Mars CMB: density_above_CMB=3800 (lower mantle), density_below_CMB=6200 (core)
     *
     * The gravitational potential perturbation must be evaluated separately
     * at the surface and CMB. At spherical harmonic degree l:
     *
     *   Phi_s = 4*pi*G/(2l+1) *
     *           [delta_rho_s*h_s*R + delta_rho_b*h_b*Rb*(Rb/R)^(l+1)]
     *   Phi_b = 4*pi*G/(2l+1) *
     *           [delta_rho_s*h_s*R*(Rb/R)^l + delta_rho_b*h_b*Rb]
     *
     * The surface traction uses Phi_s. The fluid-core CMB traction uses
     * delta_rho_cmb * (g*h_cmb - Phi_b), after subtracting the mantle
     * reference hydrostatic state. The two operators are not interchangeable.
     *
     * Usage in input file:
     * @code
     * subsection Boundary traction model
     *   set Prescribed traction boundary indicators = top: self gravitation
     *   subsection Self gravitation
     *     set Maximum degree            = 40
     *     set Density above surface     = 0      # vacuum/atmosphere
     *     set Density below surface     = 3500   # crust
     *     set Density above CMB         = 3800   # lower mantle
     *     set Density below CMB         = 6200   # core
     *     set Include CMB contribution  = true
     *   end
     * end
     * @endcode
     */
    template <int dim>
    class SelfGravitation : public BoundaryTraction::Interface<dim>,
      public ::aspect::SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        void update() override;

        Tensor<1,dim>
        boundary_traction(const types::boundary_id boundary_indicator,
                          const Point<dim> &position,
                          const Tensor<1,dim> &normal_vector) const override;

        /** Return the current non-local potential height Phi/g. */
        double potential_height(const types::boundary_id boundary_indicator,
                                const Point<dim> &position) const;

        /** Add a load owned by the unified potential-feedback adapter. */
        void set_additional_load_traction_function(
          const std::function<Tensor<1,dim>(const types::boundary_id,
                                            const Point<dim> &,
                                            const Tensor<1,dim> &)> &function);

        /** Return the cosine/sine coefficient of Phi/g at the surface due
         * to the effective surface mass (external load plus topography). */
        std::pair<double,double>
        surface_mass_potential_coefficient(const unsigned int degree,
                                           const unsigned int order) const;

        /** Return the cosine/sine coefficient of Phi/g at the surface due
         * to the externally applied surface load before degree-1 reference
         * frame cancellation. */
        std::pair<double,double>
        external_load_surface_potential_coefficient(const unsigned int degree,
                                                    const unsigned int order) const;

        /** Return the cosine/sine coefficient of Phi/g at the surface due
         * to surface deformation before degree-1 reference-frame
         * cancellation. */
        std::pair<double,double>
        surface_deformation_mass_potential_coefficient(const unsigned int degree,
                                                       const unsigned int order) const;

        /** Return the cosine/sine coefficient of Phi/g at the surface due
         * to CMB topography. These accessors let geoid output use the same
         * converged boundary state as the traction operator. */
        std::pair<double,double>
        cmb_mass_potential_coefficient(const unsigned int degree,
                                       const unsigned int order) const;

        /** Return the cosine/sine coefficient of the externally applied
         * tidal potential height, Phi/g, evaluated at the surface. */
        std::pair<double,double>
        tidal_surface_potential_coefficient(const unsigned int degree,
                                            const unsigned int order) const;

        /** Return the cosine/sine coefficient of the reference-frame
         * correction to Phi/g evaluated at the surface. */
        std::pair<double,double>
        reference_frame_surface_potential_coefficient(const unsigned int degree,
                                                      const unsigned int order) const;

        /** Return the uniform acceleration associated with the degree-1
         * center-of-mass reference-frame correction. */
        Tensor<1,dim>
        reference_frame_body_force(const Point<dim> &position) const;

        /** Return the center-of-mass displacement vector computed from the
         * total degree-1 surface + CMB mass potential, including any external
         * surface load. */
        Tensor<1,dim>
        get_cm_displacement_increment() const;

        /** Return the center-of-mass displacement vector computed from the
         * deformation-only degree-1 surface + CMB mass potential, excluding
         * externally applied surface loads. */
        Tensor<1,dim>
        get_deformation_cm_displacement_increment() const;

        /** Return the center-of-mass displacement vector computed from the
         * externally applied degree-1 surface-load mass potential only. */
        Tensor<1,dim>
        get_external_load_cm_displacement_increment() const;

        /** Return the center-of-mass displacement vector computed from the
         * degree-1 surface-deformation mass potential only. */
        Tensor<1,dim>
        get_surface_deformation_cm_displacement_increment() const;

        /** Return the center-of-mass displacement vector computed from the
         * degree-1 CMB-deformation mass potential only. */
        Tensor<1,dim>
        get_cmb_deformation_cm_displacement_increment() const;

        /** Return the density jump used for the surface mass term. */
        double surface_density_jump() const;

        /** Return the density jump used for the CMB mass term. */
        double cmb_density_jump() const;

        bool has_citcomsve_degree_one_load_replay_diagnostic() const;
        double citcomsve_degree_one_cmb_intermediate_compensation_rhs_10() const;
        double citcomsve_degree_one_cmb_potential_append_rhs_10() const;
        double citcomsve_degree_one_cmb_final_rhs_10() const;

        /** Whether the last post-Stokes update changed the combined surface
         * and CMB Phi/g coefficient vectors by less than the configured
         * relative tolerance. */
        bool potential_is_converged() const;

        double potential_relative_change_value() const;

        unsigned int minimum_degree() const;

        /**
         * Return the current self-gravitational potential Phi at @p position.
         * The cached field is available for the 3-D mechanical mass-
         * conservation formulation and is zero otherwise.
         */
        double
        full_domain_potential(const Point<dim> &position) const;

        /** Return whether a full-domain self-gravity potential is cached. */
        bool
        has_full_domain_potential() const;

        void configure_from_potential_feedback_settings(
          const PotentialFeedback::Settings &settings);

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

        std::pair<std::vector<double>, std::vector<double>>
        compute_internal_density_potential(const double outer_radius) const;

        std::pair<std::pair<double, std::pair<std::vector<double>, std::vector<double>>>, std::pair<double, std::pair<std::vector<double>, std::vector<double>>>>
        compute_topography_potential(const double outer_radius, const double inner_radius) const;

        std::string get_include_internal_density_anomalies() const;
        double get_reference_density_for_internal_anomalies() const;
        double get_internal_density_anomaly_tolerance() const;

      private:
        enum class AnalysisBoundary
        {
          surface,
          cmb
        };

        struct CitcomSVEDegreeOneLoadReplayDiagnostic
        {
          bool valid = false;
          double original_surface_load_height_10 = 0.0;
          double phi_external_10_over_g = 0.0;
          double citcomsve_cm_z = 0.0;
          double citcomsve_h_comp_10 = 0.0;
          double corrected_surface_load_height_10 = 0.0;
          double corrected_cmb_load_height_10 = 0.0;
          double surface_kernel_l1 = 0.0;
          double cmb_kernel_l1 = 0.0;
          double net_degree1_phi_over_g_after_load_compensation = 0.0;
          double phi_cmb_pre_cancellation_over_g_10 = 0.0;
          double h_comp_10 = 0.0;
          double surface_deformation_topo_cos_10 = 0.0;
          double cmb_deformation_topo_cos_10 = 0.0;
          double surface_to_cmb_l1 = 0.0;
          double cmb_to_cmb_l1 = 0.0;
          double surface_deformation_to_cmb_phi_over_g_10 = 0.0;
          double cmb_deformation_to_cmb_phi_over_g_10 = 0.0;
          double original_surface_load_to_cmb_l1_times_height = 0.0;
          double surface_to_cmb_l1_times_h_comp = 0.0;
          double cmb_to_cmb_l1_times_h_comp = 0.0;
          double phi_cmb_deformation_pre_compensation_over_g_10 = 0.0;
          double phi_cmb_initial_load_pair_replay_over_g_10 = 0.0;
          double phi_cmb_replay_over_g_10 = 0.0;
          double cmb_intermediate_compensation_rhs_10 = 0.0;
          double cmb_potential_append_rhs_10 = 0.0;
          double cmb_final_rhs_10 = 0.0;
        };

        struct NativeCenterOfMassDiagnostic
        {
          bool valid = false;
          Tensor<1,3> mass_dipole_pre;
          Tensor<1,3> mass_dipole_post;
          Tensor<1,3> translation;
          Tensor<1,3> internal_density_dipole;
          Tensor<1,3> surface_interface_dipole;
          Tensor<1,3> cmb_interface_dipole;
          Tensor<1,3> external_load_dipole;
          double total_mass = 0.0;
          double correctable_mass = 0.0;
        };

        /**
         * Compute the self-gravity correction for the current topography.
         * Collects surface topography from mesh deformation, performs SH
         * analysis, applies the self-gravity kernel, and synthesizes the
         * correction field. Stores results for use by boundary_traction().
         */
        void
        compute_self_gravity_correction(
          const bool include_current_velocity_increment);

        /** Update the non-local boundary operator after a Stokes solve so
         * that the next nonlinear iteration uses the current displacement
         * estimate, rather than lagging the feedback by a full time step. */
        void update_after_stokes_solve();

        std::pair<std::vector<double>, std::vector<double>>
        to_spherical_harmonic_coefficients(const std::vector<std::vector<double>> &spherical_function) const;

        std::vector<std::pair<std::vector<double>, std::vector<double>>>
        timed_spherical_harmonic_analysis_multiple(
          const std::vector<double> &theta,
          const std::vector<double> &phi,
          const std::vector<double> &weights,
          const std::vector<std::vector<double>> &values,
          const MPI_Comm &mpi_comm,
          const AnalysisBoundary analysis_boundary) const;

        std::vector<double>
        timed_spherical_harmonic_synthesis(
          const std::vector<double> &cos_coeffs,
          const std::vector<double> &sin_coeffs,
          const std::vector<double> &theta,
          const std::vector<double> &phi) const;

        Tensor<1,3>
        degree_one_mass_dipole_from_height_coefficients(
          const std::vector<double> &cos_coeffs,
          const std::vector<double> &sin_coeffs,
          const double density_jump,
          const double radius) const;

        Tensor<1,3>
        compute_internal_density_mass_dipole() const;

        /**
         * Cache Phi/g_surface spherical-harmonic coefficients on radial
         * support points for compressible full-domain potential forcing.
         */
        void
        update_full_domain_potential(
          const std::vector<double> &surface_height_cos,
          const std::vector<double> &surface_height_sin,
          const std::vector<double> &cmb_height_cos,
          const std::vector<double> &cmb_height_sin,
          const double outer_radius,
          const double inner_radius);

        void
        write_native_center_of_mass_diagnostic(
          const bool include_current_velocity_increment) const;

        void
        update_derived_planetary_constants();

        unsigned int max_degree;
        unsigned int min_degree;

        double density_above_surface;
        double density_below_surface;
        double density_above_cmb;
        double density_below_cmb;
        double planet_mean_density = 0.0;
        double planet_mass = 0.0;
        bool   include_cmb_contribution;
        bool   include_surface_contribution;
        bool   self_gravity_mass_feedback_enabled;
        bool   iterate_with_stokes;
        bool   freeze_potential_after_timestep_zero;
        double initial_displacement_timestep;
        double potential_convergence_tolerance;
        double potential_iteration_relaxation_factor;
        double potential_relative_change;
        unsigned int maximum_potential_iterations;
        unsigned int current_potential_iteration_step;
        unsigned int potential_iteration_number;
        bool   enable_surface_potential_traction;
        bool   enable_cmb_potential_traction;
        DegreeOneReferenceFrame degree_one_reference_frame;
        bool   center_of_mass_correction;
        bool   citcomsve_degree_one_load_compensation;
        TidalPotential tidal_potential;

        std::string include_internal_density_anomalies;
        double reference_density_for_internal_anomalies;
        double internal_density_anomaly_tolerance;
        std::string full_domain_volume_source_discretization = "quadrature point";
        unsigned int full_domain_potential_radial_subdivisions = 32;

        double time_between_text_output;
        unsigned int time_steps_between_text_output;

        mutable double last_text_output_time;
        mutable unsigned int last_text_output_step;
        mutable unsigned int current_tracked_step;
        mutable bool printing_this_step;

        types::boundary_id top_boundary_id;
        types::boundary_id bottom_boundary_id;

        bool configured_from_potential_feedback = false;

        std::function<Tensor<1,dim>(const types::boundary_id,
                                    const Point<dim> &,
                                    const Tensor<1,dim> &)>
        additional_load_traction_function;

        /**
         * The SH transform utility (3D) or Fourier transform (2D).
         */
        std::unique_ptr<Utilities::SphericalHarmonicTransform> sh_transform;
        std::unique_ptr<Utilities::FourierTransform> fourier_transform;
        mutable Utilities::SphericalHarmonicBasisCache surface_analysis_basis_cache;
        mutable Utilities::SphericalHarmonicBasisCache cmb_analysis_basis_cache;

        // Coefficients of Phi/g evaluated at each boundary.
        std::vector<double> surface_potential_cos_coeffs;
        std::vector<double> surface_potential_sin_coeffs;
        std::vector<double> cmb_potential_cos_coeffs;
        std::vector<double> cmb_potential_sin_coeffs;

        // Phi/g_surface coefficients on radial support points. These vectors
        // contain the self-gravity mass potential only; tidal, rotational,
        // and reference-frame potentials remain separate.
        std::vector<double> full_domain_potential_radii;
        std::vector<std::vector<double>> full_domain_potential_cos_coeffs;
        std::vector<std::vector<double>> full_domain_potential_sin_coeffs;
        double full_domain_reference_gravity = 0.0;

        // Separate contributions to Phi/g at the outer surface. Their sum is
        // surface_potential_*; retaining the split avoids reconstructing
        // predicted ALE topography from total boundary traction in the geoid
        // postprocessor.
        std::vector<double> surface_mass_potential_cos_coeffs;
        std::vector<double> surface_mass_potential_sin_coeffs;
        std::vector<double> external_load_surface_potential_cos_coeffs;
        std::vector<double> external_load_surface_potential_sin_coeffs;
        std::vector<double> surface_deformation_mass_potential_cos_coeffs;
        std::vector<double> surface_deformation_mass_potential_sin_coeffs;
        std::vector<double> cmb_mass_potential_cos_coeffs;
        std::vector<double> cmb_mass_potential_sin_coeffs;

        // Externally prescribed tidal potential, Phi/g, evaluated at the
        // surface and CMB. This is separate from
        // the self-gravitational potential generated by boundary mass.
        std::vector<double> tidal_surface_potential_cos_coeffs;
        std::vector<double> tidal_surface_potential_sin_coeffs;
        std::vector<double> tidal_cmb_potential_cos_coeffs;
        std::vector<double> tidal_cmb_potential_sin_coeffs;

        // Reference-frame potential that enforces the degree-1 center-of-mass
        // convention without erasing the physical surface/CMB mass
        // coefficients used for displacement diagnostics.
        std::vector<double> reference_frame_surface_potential_cos_coeffs;
        std::vector<double> reference_frame_surface_potential_sin_coeffs;
        std::vector<double> reference_frame_cmb_potential_cos_coeffs;
        std::vector<double> reference_frame_cmb_potential_sin_coeffs;
        Tensor<1,dim> reference_frame_acceleration;

        // Degree-1 equivalent topography coefficients for the
        // CitcomSVE-style center-of-mass compensating load.
        std::vector<double> degree_one_load_compensation_cos_coeffs;
        std::vector<double> degree_one_load_compensation_sin_coeffs;
        std::vector<double> degree_one_load_replay_cmb_potential_cos_coeffs;
        std::vector<double> degree_one_load_replay_cmb_potential_sin_coeffs;

        // CMB topography coefficients used by the direct density-jump
        // restoring traction.
        std::vector<double> cmb_topography_cos_coeffs;
        std::vector<double> cmb_topography_sin_coeffs;

        // Committed CMB topography. These coefficients are retained separately
        // from cmb_topography_* because the CMB direct density-jump traction
        // can be evaluated either from the committed state, from the current
        // post-Stokes topography estimate, or omitted entirely in diagnostic
        // Zhong et al. (2022) benchmark experiments.
        std::vector<double> cmb_committed_topography_cos_coeffs;
        std::vector<double> cmb_committed_topography_sin_coeffs;

        // Center-of-mass displacement vectors (meters) inferred from degree-1
        // mass-potential coefficients.
        Tensor<1,dim> cm_displacement_increment;
        Tensor<1,dim> deformation_cm_displacement_increment;
        Tensor<1,dim> external_load_cm_displacement_increment;
        Tensor<1,dim> surface_deformation_cm_displacement_increment;
        Tensor<1,dim> cmb_deformation_cm_displacement_increment;

        CitcomSVEDegreeOneLoadReplayDiagnostic
        citcomsve_degree_one_load_replay_diagnostic;
        NativeCenterOfMassDiagnostic native_center_of_mass_diagnostic;
    };
  }
}

#endif
