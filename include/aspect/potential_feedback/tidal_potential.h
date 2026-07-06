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

#ifndef _aspect_potential_feedback_tidal_potential_h
#define _aspect_potential_feedback_tidal_potential_h

#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>

#include <string>
#include <vector>

namespace aspect
{
  namespace PotentialFeedback
  {
    struct Settings;

    /**
     * Helper for externally prescribed tidal potentials. The helper stores
     * the prescribed Phi/g coefficient at the surface and maps it to the CMB.
     * SelfGravitation owns the actual boundary traction path, because the
     * tidal potential acts through the same density-interface operator as the
     * self-gravity potential.
     */
    class TidalPotential
    {
      public:
        static void declare_parameters(ParameterHandler &prm);

        void configure_from_settings(const Settings &settings,
                                     const unsigned int min_degree,
                                     const unsigned int max_degree,
                                     const unsigned int dimension);

        void add_to_coefficients(
          const Utilities::SphericalHarmonicTransform &sh_transform,
          const double radius_ratio,
          const double time,
          std::vector<double> &surface_potential_cos_coeffs,
          std::vector<double> &surface_potential_sin_coeffs,
          std::vector<double> &cmb_potential_cos_coeffs,
          std::vector<double> &cmb_potential_sin_coeffs,
          std::vector<double> &tidal_surface_potential_cos_coeffs,
          std::vector<double> &tidal_surface_potential_sin_coeffs,
          std::vector<double> &tidal_cmb_potential_cos_coeffs,
          std::vector<double> &tidal_cmb_potential_sin_coeffs) const;

      private:
        bool enabled = false;
        unsigned int degree = 2;
        unsigned int order = 0;
        double potential_height_amplitude = 0.0;
        double potential_amplitude = 0.0;
        double reference_gravity = 1.0;
        std::string coefficient_type = "cosine";
        std::string normalization = "geodesy 4pi";
        std::string potential_quantity = "potential height";
        std::string time_dependence = "none";
        double angular_frequency = 0.0;
        double phase = 0.0;
    };
  }
}

#endif
