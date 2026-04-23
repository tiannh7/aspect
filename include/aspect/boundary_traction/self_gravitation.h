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
     * The gravitational potential perturbation at spherical harmonic degree l:
     *
     *   delta_Phi_l = (4*pi*G*R/(2l+1)) * [ delta_rho_surf * h_surf_l
     *                                       + delta_rho_cmb * h_cmb_l * (r_cmb/R)^(l+2) ]
     *
     * The self-gravity ratio per degree (surface contribution):
     *
     *   Rsg_surf_l = 3 * delta_rho_surf / ((2l+1) * rho_mean)
     *
     * The CMB contribution (applied as correction at the surface):
     *
     *   Rsg_cmb_l = 3 * delta_rho_cmb / ((2l+1) * rho_mean) * (r_cmb/R)^(l+2)
     *
     * These are applied as outward normal traction corrections on the
     * respective boundaries, reducing the effective load by accounting for
     * the gravitational attraction of the deformed surfaces.
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

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

      private:
        /**
         * Compute the self-gravity correction for the current topography.
         * Collects surface topography from mesh deformation, performs SH
         * analysis, applies the self-gravity kernel, and synthesizes the
         * correction field. Stores results for use by boundary_traction().
         */
        void compute_self_gravity_correction();

        unsigned int max_degree;
        unsigned int min_degree;

        double density_above_surface;
        double density_below_surface;
        double density_above_cmb;
        double density_below_cmb;
        double planet_mean_density;
        bool   include_cmb_contribution;

        /**
         * The SH transform utility (3D) or Fourier transform (2D).
         */
        std::unique_ptr<Utilities::SphericalHarmonicTransform> sh_transform;
        std::unique_ptr<Utilities::FourierTransform> fourier_transform;

        /**
         * Cached data from the last update():
         * positions, normals, and the self-gravity correction values
         * at surface quadrature points.
         */
        std::vector<double> cached_theta;
        std::vector<double> cached_phi;
        std::vector<double> cached_correction;

        /**
         * SH coefficients of the total self-gravity correction
         * (surface + CMB contributions combined, for synthesis
         * at arbitrary points on the surface boundary).
         */
        std::vector<double> correction_cos_coeffs;
        std::vector<double> correction_sin_coeffs;
    };
  }
}

#endif
