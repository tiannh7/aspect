/*
  Copyright (C) 2026 by the authors of the ASPECT code.

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

#ifndef _aspect_boundary_traction_applied_potential_h
#define _aspect_boundary_traction_applied_potential_h

#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>

#include <string>
#include <vector>

namespace aspect
{
  namespace BoundaryTraction
  {
    /**
     * Helper for externally applied gravitational potentials, such as the
     * degree-2 tidal potential used by Zhong et al. (2022). The helper stores
     * the prescribed Phi/g coefficient at the surface and maps it to the CMB
     * with a configurable radial power. SelfGravitation owns the actual
     * boundary traction path, because the applied potential acts through the
     * same density-interface operator as the self-gravity potential.
     */
    class AppliedPotential
    {
      public:
        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm,
                              const unsigned int min_degree,
                              const unsigned int max_degree,
                              const unsigned int dimension);

        void add_to_coefficients(
          const Utilities::SphericalHarmonicTransform &sh_transform,
          const double radius_ratio,
          std::vector<double> &surface_potential_cos_coeffs,
          std::vector<double> &surface_potential_sin_coeffs,
          std::vector<double> &cmb_potential_cos_coeffs,
          std::vector<double> &cmb_potential_sin_coeffs,
          std::vector<double> &applied_surface_potential_cos_coeffs,
          std::vector<double> &applied_surface_potential_sin_coeffs,
          std::vector<double> &applied_cmb_potential_cos_coeffs,
          std::vector<double> &applied_cmb_potential_sin_coeffs) const;

      private:
        bool enabled = false;
        unsigned int degree = 2;
        unsigned int order = 0;
        double surface_height = 0.0;
        double radial_exponent = 2.0;
        double sign = 1.0;
        std::string coefficient_type = "cosine";
        std::string normalization = "geodesy 4pi";
    };
  }
}

#endif
