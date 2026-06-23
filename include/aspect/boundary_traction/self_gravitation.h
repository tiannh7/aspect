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

#ifndef _aspect_boundary_traction_self_gravitation_h
#define _aspect_boundary_traction_self_gravitation_h

#include <aspect/boundary_traction/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <memory>

namespace aspect
{
  namespace BoundaryTraction
  {
    /**
     * A boundary traction plugin that computes the self-gravitational
     * feedback from surface topography. Supports both 3D (spherical shell,
     * using spherical harmonic expansion) and 2D (annulus, using Fourier
     * expansion).
     *
     * The density contrasts at the surface and CMB are:
     *   delta_rho_surf = density_below_surface - density_above_surface
     *   delta_rho_cmb  = density_below_cmb - density_above_cmb
     *
     * For example (surface):
     *   - Rock topography in vacuum: density_below_surface=3500, density_above_surface=0
     *   - Ice cap on rock: density_below_surface=917, density_above_surface=0
     * For example (CMB):
     *   - Earth CMB: density_above_cmb=5500 (lower mantle), density_below_cmb=9900 (outer core)
     *   - Mars CMB: density_above_cmb=3800 (lower mantle), density_below_cmb=6200 (core)
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
     *     set Density above cmb         = 3800   # lower mantle
     *     set Density below cmb         = 6200   # core
     *     set Planet mean density       = 3390   # Mars
     *     set Include cmb contribution  = true
     *   end
     * end
     * @endcode
     */
    template <int dim>
    class SelfGravitation : public Interface<dim>,
      public ::aspect::SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        void update() override;

        Tensor<1,dim>
        boundary_traction(const types::boundary_id boundary_indicator,
                          const Point<dim> &position,
                          const Tensor<1,dim> &normal_vector) const override;

        /** Return the cosine/sine coefficient of Phi/g at the surface due
         * to the effective surface mass (external load plus topography). */
        std::pair<double,double>
        surface_mass_potential_coefficient(const unsigned int degree,
                                           const unsigned int order) const;

        /** Return the cosine/sine coefficient of Phi/g at the surface due
         * to CMB topography. These accessors let geoid output use the same
         * converged boundary state as the traction operator. */
        std::pair<double,double>
        cmb_mass_potential_coefficient(const unsigned int degree,
                                       const unsigned int order) const;

        /** Whether the last post-Stokes update changed the combined surface
         * and CMB Phi/g coefficient vectors by less than the configured
         * relative tolerance. */
        bool potential_is_converged() const;

        double potential_relative_change_value() const;

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

      private:
        /**
         * Compute the self-gravity correction for the current topography.
         * Collects surface topography from mesh deformation, performs SH
         * analysis, applies the self-gravity kernel, and synthesizes the
         * correction field. Stores results for use by boundary_traction().
         */
        void compute_self_gravity_correction(const bool include_current_velocity_increment);

        /** Update the non-local boundary operator after a Stokes solve so
         * that the next nonlinear iteration uses the current displacement
         * estimate, rather than lagging the feedback by a full time step. */
        void update_after_stokes_solve();

        unsigned int max_degree;
        unsigned int min_degree;

        double density_above_surface;
        double density_below_surface;
        double density_above_cmb;
        double density_below_cmb;
        double planet_mean_density;
        bool   include_cmb_contribution;
        bool   iterate_with_stokes;
        double initial_displacement_timestep;
        double potential_convergence_tolerance;
        double potential_relative_change;
        double cmb_potential_traction_sign;

        types::boundary_id top_boundary_id;
        types::boundary_id bottom_boundary_id;

        /**
         * The SH transform utility (3D) or Fourier transform (2D).
         */
        std::unique_ptr<Utilities::SphericalHarmonicTransform> sh_transform;
        std::unique_ptr<Utilities::FourierTransform> fourier_transform;

        // Coefficients of Phi/g evaluated at each boundary.
        std::vector<double> surface_potential_cos_coeffs;
        std::vector<double> surface_potential_sin_coeffs;
        std::vector<double> cmb_potential_cos_coeffs;
        std::vector<double> cmb_potential_sin_coeffs;

        // Separate contributions to Phi/g at the outer surface. Their sum is
        // surface_potential_*; retaining the split avoids reconstructing
        // predicted ALE topography from total boundary traction in the geoid
        // postprocessor.
        std::vector<double> surface_mass_potential_cos_coeffs;
        std::vector<double> surface_mass_potential_sin_coeffs;
        std::vector<double> cmb_mass_potential_cos_coeffs;
        std::vector<double> cmb_mass_potential_sin_coeffs;

        // CMB topography coefficients used by the direct density-jump
        // restoring traction.
        std::vector<double> cmb_topography_cos_coeffs;
        std::vector<double> cmb_topography_sin_coeffs;

        // Committed CMB topography. The current displacement increment is
        // represented implicitly in the free-boundary restoring matrix and
        // must not be duplicated on the traction RHS.
        std::vector<double> cmb_committed_topography_cos_coeffs;
        std::vector<double> cmb_committed_topography_sin_coeffs;
    };
  }
}

#endif
