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

#include <aspect/boundary_traction/tidal_potential.h>

#include <cmath>

namespace aspect
{
  namespace BoundaryTraction
  {
    void
    TidalPotential::declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry("Enable", "false",
                        Patterns::Bool(),
                        "Whether to add an externally prescribed tidal "
                        "potential to the boundary-potential traction. The "
                        "potential is prescribed as Phi/g at the outer surface "
                        "and evaluated separately at the surface and CMB.");
      prm.declare_entry("Degree", "2",
                        Patterns::Integer(0),
                        "Spherical harmonic degree of the tidal potential.");
      prm.declare_entry("Order", "0",
                        Patterns::Integer(0),
                        "Spherical harmonic order of the tidal potential.");
      prm.declare_entry("Coefficient type", "cosine",
                        Patterns::Selection("cosine|sine"),
                        "Select the real spherical harmonic coefficient type "
                        "for the tidal potential.");
      prm.declare_entry("Normalization",
                        "geodesy 4pi",
                        Patterns::Selection("geodesy 4pi|unnormalized legendre"),
                        "Normalization of the tidal potential. The "
                        "'geodesy 4pi' option uses ASPECT's "
                        "Utilities::real_spherical_harmonic convention. The "
                        "'unnormalized legendre' option prescribes "
                        "P_l(cos theta) and is restricted to m=0.");
      prm.declare_entry("Height at surface", "0",
                        Patterns::Double(),
                        "Amplitude of the tidal potential at the outer "
                        "surface in height units, Phi/g, multiplying the "
                        "selected angular function. Units are meters.");
      prm.declare_entry("Radial exponent", "2",
                        Patterns::Double(),
                        "Power p in Phi(r)/g = Phi(R)/g * (r/R)^p. Zhong "
                        "et al. (2022) degree-2 tidal forcing uses p=2.");
      prm.declare_entry("Sign", "1",
                        Patterns::Double(),
                        "Diagnostic multiplier on the tidal potential "
                        "amplitude.");
    }



    void
    TidalPotential::parse_parameters(ParameterHandler &prm,
                                     const unsigned int min_degree,
                                     const unsigned int max_degree,
                                     const unsigned int dimension)
    {
      enabled = prm.get_bool("Enable");
      degree = prm.get_integer("Degree");
      order = prm.get_integer("Order");
      coefficient_type = prm.get("Coefficient type");
      normalization = prm.get("Normalization");
      surface_height = prm.get_double("Height at surface");
      radial_exponent = prm.get_double("Radial exponent");
      sign = prm.get_double("Sign");

      AssertThrow(order <= degree,
                  ExcMessage("Tidal potential order must not exceed its "
                             "degree."));
      if (enabled)
        {
          AssertThrow(dimension == 3,
                      ExcMessage("Tidal potential forcing is currently "
                                 "implemented only for 3D spherical shells."));
          AssertThrow(degree >= min_degree && degree <= max_degree,
                      ExcMessage("Tidal potential degree must lie between "
                                 "Minimum degree and Maximum degree."));
          if (order == 0)
            AssertThrow(coefficient_type == "cosine",
                        ExcMessage("The sine coefficient is zero for m=0. "
                                   "Use 'Coefficient type = cosine'."));
          if (normalization == "unnormalized legendre")
            AssertThrow(order == 0,
                        ExcMessage("The 'unnormalized legendre' tidal "
                                   "potential normalization is only "
                                   "implemented for m=0."));
        }
    }



    void
    TidalPotential::add_to_coefficients(
      const Utilities::SphericalHarmonicTransform &sh_transform,
      const double radius_ratio,
      std::vector<double> &surface_potential_cos_coeffs,
      std::vector<double> &surface_potential_sin_coeffs,
      std::vector<double> &cmb_potential_cos_coeffs,
      std::vector<double> &cmb_potential_sin_coeffs,
      std::vector<double> &tidal_surface_potential_cos_coeffs,
      std::vector<double> &tidal_surface_potential_sin_coeffs,
      std::vector<double> &tidal_cmb_potential_cos_coeffs,
      std::vector<double> &tidal_cmb_potential_sin_coeffs) const
    {
      if (!enabled)
        return;

      const unsigned int i_applied = sh_transform.index(degree, order);
      double coefficient = sign * surface_height;

      if (normalization == "unnormalized legendre")
        {
          const std::pair<double,double> pole_value =
            Utilities::real_spherical_harmonic(degree, order, 0.0, 0.0);
          AssertThrow(std::abs(pole_value.first) > 0.0,
                      ExcMessage("Cannot convert the requested unnormalized "
                                 "Legendre tidal potential to the internal "
                                 "spherical-harmonic normalization."));
          coefficient /= pole_value.first;
        }

      const double cmb_coefficient =
        coefficient * std::pow(radius_ratio, radial_exponent);

      if (coefficient_type == "cosine")
        {
          tidal_surface_potential_cos_coeffs[i_applied] = coefficient;
          tidal_cmb_potential_cos_coeffs[i_applied] = cmb_coefficient;
        }
      else
        {
          tidal_surface_potential_sin_coeffs[i_applied] = coefficient;
          tidal_cmb_potential_sin_coeffs[i_applied] = cmb_coefficient;
        }

      for (unsigned int i = 0; i < surface_potential_cos_coeffs.size(); ++i)
        {
          surface_potential_cos_coeffs[i] +=
            tidal_surface_potential_cos_coeffs[i];
          surface_potential_sin_coeffs[i] +=
            tidal_surface_potential_sin_coeffs[i];
          cmb_potential_cos_coeffs[i] +=
            tidal_cmb_potential_cos_coeffs[i];
          cmb_potential_sin_coeffs[i] +=
            tidal_cmb_potential_sin_coeffs[i];
        }
    }
  }
}
