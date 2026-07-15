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

#include <aspect/potential_feedback/tidal_potential.h>
#include <aspect/potential_feedback/interface.h>

#include <cmath>

namespace aspect
{
  namespace PotentialFeedback
  {
    void
    TidalPotential::declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry("Model name", "none",
                        Patterns::Selection("none|spherical harmonic potential"),
                        "Select the tidal-potential model. The `none' model "
                        "disables externally prescribed tidal potential. The "
                        "`spherical harmonic potential' model adds one real "
                        "spherical-harmonic Phi/g coefficient directly to the "
                        "potential-feedback coefficient arrays.");

      prm.enter_subsection("Spherical harmonic potential");
      {
        prm.declare_entry("Harmonic degree", "2",
                          Patterns::Integer(0),
                          "Spherical harmonic degree of the tidal potential.");
        prm.declare_entry("Harmonic order", "0",
                          Patterns::Integer(0),
                          "Spherical harmonic order of the tidal potential.");
        prm.declare_entry("Coefficient type", "cosine",
                          Patterns::Selection("cosine|sine"),
                          "Select the real spherical harmonic coefficient type "
                          "for the tidal potential.");
        prm.declare_entry("Potential quantity", "potential height",
                          Patterns::Selection("potential height|potential"),
                          "Whether the input amplitude is Phi/g in meters or "
                          "Phi in m^2/s^2.");
        prm.declare_entry("Potential height amplitude", "0.0",
                          Patterns::Double(),
                          "Amplitude of Phi/g at the outer surface in meters.");
        prm.declare_entry("Potential amplitude", "0.0",
                          Patterns::Double(),
                          "Amplitude of Phi at the outer surface in m^2/s^2. "
                          "Used when Potential quantity is `potential'.");
        prm.declare_entry("Reference gravity", "1.0",
                          Patterns::Double(0.0),
                          "Reference gravity used to convert Phi to Phi/g.");
        prm.declare_entry("Normalization", "geodesy 4pi",
                          Patterns::Selection("geodesy 4pi|unnormalized legendre"),
                          "Normalization of the tidal potential. The "
                          "`geodesy 4pi' option uses ASPECT's "
                          "Utilities::real_spherical_harmonic convention. The "
                          "`unnormalized legendre' option prescribes "
                          "P_l(cos theta) and is restricted to m=0.");
        prm.declare_entry("Time dependence", "none",
                          Patterns::Selection("none|sinusoidal"),
                          "Whether the tidal potential is static or multiplied "
                          "by cos(Angular frequency * time + Phase).");
        prm.declare_entry("Angular frequency", "0.0",
                          Patterns::Double(),
                          "Angular frequency for sinusoidal time dependence.");
        prm.declare_entry("Phase", "0.0",
                          Patterns::Double(),
                          "Phase for sinusoidal time dependence in radians.");
      }
      prm.leave_subsection();
    }



    void
    TidalPotential::configure_from_settings(const Settings &settings,
                                            const unsigned int min_degree,
                                            const unsigned int max_degree,
                                            const unsigned int dimension)
    {
      enabled = settings.tidal_model_name == "spherical harmonic potential";
      degree = settings.tidal_harmonic_degree;
      order = settings.tidal_harmonic_order;
      coefficient_type = settings.tidal_coefficient_type;
      potential_quantity = settings.tidal_potential_quantity;
      potential_height_amplitude = settings.tidal_potential_height_amplitude;
      potential_amplitude = settings.tidal_potential_amplitude;
      reference_gravity = settings.tidal_reference_gravity;
      normalization = settings.tidal_normalization;
      time_dependence = settings.tidal_time_dependence;
      angular_frequency = settings.tidal_angular_frequency;
      phase = settings.tidal_phase;

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
          if (potential_quantity == "potential")
            AssertThrow(reference_gravity > 0.0,
                        ExcMessage("Reference gravity must be positive when "
                                   "tidal Potential quantity is `potential'."));
        }
    }



    void
    TidalPotential::add_to_coefficients(
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
      std::vector<double> &tidal_cmb_potential_sin_coeffs) const
    {
      if (!enabled)
        return;

      const unsigned int i_applied = sh_transform.index(degree, order);
      const double coefficient = surface_potential_height_coefficient(time);

      const double cmb_coefficient =
        coefficient * std::pow(radius_ratio, static_cast<int>(degree));

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



    bool
    TidalPotential::is_enabled() const
    {
      return enabled;
    }



    double
    TidalPotential::full_domain_potential_height(const Point<3> &position,
                                                 const double outer_radius,
                                                 const double time) const
    {
      if (!enabled)
        return 0.0;

      AssertThrow(outer_radius > 0.0,
                  ExcMessage("The tidal-potential outer radius must be positive."));

      const double radius = position.norm();
      AssertThrow(radius > 0.0,
                  ExcMessage("The tidal potential is undefined at radius zero."));

      const std::array<double,3> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      const std::pair<double,double> harmonic =
        Utilities::real_spherical_harmonic(degree,
                                           order,
                                           spherical_coordinates[2],
                                           spherical_coordinates[1]);
      const double selected_harmonic =
        (coefficient_type == "cosine" ? harmonic.first : harmonic.second);

      return surface_potential_height_coefficient(time)
             * std::pow(radius / outer_radius, static_cast<int>(degree))
             * selected_harmonic;
    }



    double
    TidalPotential::surface_potential_height_coefficient(const double time) const
    {
      double coefficient =
        (potential_quantity == "potential height"
         ? potential_height_amplitude
         : potential_amplitude / reference_gravity);

      if (time_dependence == "sinusoidal")
        coefficient *= std::cos(angular_frequency * time + phase);

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

      return coefficient;
    }
  }
}
