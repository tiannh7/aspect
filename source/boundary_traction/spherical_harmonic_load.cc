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


#include <aspect/boundary_traction/spherical_harmonic_load.h>
#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>

#include <array>
#include <cmath>

namespace aspect
{
  namespace BoundaryTraction
  {
    template <int dim>
    Tensor<1,dim>
    SphericalHarmonicLoad<dim>::
    boundary_traction (const types::boundary_id,
                       const Point<dim> &position,
                       const Tensor<1,dim> &normal_vector) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("The spherical harmonic load boundary traction "
                             "plugin is only implemented for 3D spherical "
                             "geometries."));

      const double scalar = scalar_load(position);

      Tensor<1,dim> direction_vector;
      if (positive_load_direction == "outward normal")
        {
          direction_vector = normal_vector;
        }
      else if (positive_load_direction == "inward normal")
        {
          direction_vector = -normal_vector;
        }
      else if (positive_load_direction == "outward radial")
        {
          Tensor<1,dim> radial_unit_vector;
          radial_unit_vector[0] = 1.;
          direction_vector = Utilities::Coordinates::spherical_to_cartesian_vector(radial_unit_vector,
                                                                                   position);
        }
      else if (positive_load_direction == "inward radial")
        {
          Tensor<1,dim> radial_unit_vector;
          radial_unit_vector[0] = -1.;
          direction_vector = Utilities::Coordinates::spherical_to_cartesian_vector(radial_unit_vector,
                                                                                   position);
        }
      else
        {
          AssertThrow(false, ExcInternalError());
        }

      return scalar * direction_vector;
    }



    template <int dim>
    bool
    SphericalHarmonicLoad<dim>::is_potential_feedback_load_source () const
    {
      return true;
    }



    template <int dim>
    double
    SphericalHarmonicLoad<dim>::
    scalar_load (const Point<dim> &position) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("The spherical harmonic load boundary traction "
                             "plugin is only implemented for 3D spherical "
                             "geometries."));

      const std::array<double,dim> spherical_position =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
      const double phi = spherical_position[1];
      const double theta = spherical_position[2];

      double harmonic_value = 0.;
      if (normalization == "geodesy 4pi")
        {
          const std::pair<double,double> values =
            Utilities::real_spherical_harmonic(harmonic_degree, harmonic_order, theta, phi);
          harmonic_value = (coefficient_type == "cosine" ? values.first : values.second);
        }
      else
        {
          AssertThrow(normalization == "unnormalized legendre", ExcInternalError());
          AssertThrow(harmonic_order == 0,
                      ExcMessage("The 'unnormalized legendre' normalization is "
                                 "currently implemented only for order m=0. "
                                 "Use 'geodesy 4pi' for m>0 loads."));
          AssertThrow(coefficient_type == "cosine",
                      ExcMessage("The sine coefficient is zero for m=0. Use "
                                 "'Coefficient type = cosine'."));
          harmonic_value = legendre_p(harmonic_degree, std::cos(theta));
        }

      double magnitude = load_magnitude;
      if (use_load_height)
        {
          const double gravity =
            this->get_gravity_model().gravity_vector(position).norm();
          magnitude = load_density * gravity * load_height;
        }

      return magnitude * harmonic_value;
    }



    template <int dim>
    double
    SphericalHarmonicLoad<dim>::
    legendre_p (const unsigned int degree,
                const double x) const
    {
      if (degree == 0)
        return 1.;
      if (degree == 1)
        return x;

      double p_l_minus_two = 1.;
      double p_l_minus_one = x;
      for (unsigned int l = 2; l <= degree; ++l)
        {
          const double p_l =
            ((2. * l - 1.) * x * p_l_minus_one
             - (l - 1.) * p_l_minus_two) / l;
          p_l_minus_two = p_l_minus_one;
          p_l_minus_one = p_l;
        }

      return p_l_minus_one;
    }



    template <int dim>
    void
    SphericalHarmonicLoad<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Spherical harmonic load");
        {
          prm.declare_entry ("Harmonic degree", "2",
                             Patterns::Integer(0),
                             "Spherical harmonic degree l.");
          prm.declare_entry ("Harmonic order", "0",
                             Patterns::Integer(0),
                             "Spherical harmonic order m. Currently this "
                             "plugin accepts non-negative orders and provides "
                             "separate cosine and sine coefficient types.");
          prm.declare_entry ("Coefficient type", "cosine",
                             Patterns::Selection("cosine|sine"),
                             "Select the real spherical harmonic coefficient "
                             "type. For m=0 only the cosine component is "
                             "nonzero.");
          prm.declare_entry ("Normalization", "geodesy 4pi",
                             Patterns::Selection("geodesy 4pi|unnormalized legendre"),
                             "Normalization of the angular load. The "
                             "'geodesy 4pi' option uses ASPECT's "
                             "Utilities::real_spherical_harmonic convention. "
                             "The 'unnormalized legendre' option evaluates "
                             "P_l(cos theta) and is currently restricted to "
                             "m=0.");

          prm.declare_alias ("Harmonic degree", "Degree");
          prm.declare_alias ("Harmonic order", "Order");

          prm.declare_entry ("Load magnitude", "-1.0",
                             Patterns::Double(),
                             "Positive scalar traction amplitude in Pa multiplying the "
                             "selected angular function. Alternatively, specify "
                             "Load height and Load density to derive this "
                             "traction amplitude from basic surface-load "
                             "quantities and the active gravity model.");
          prm.declare_entry ("Load height", "-1.0",
                             Patterns::Double(),
                             "Positive harmonic load-height amplitude in meters. If this "
                             "parameter is non-negative then the scalar traction "
                             "amplitude is computed as Load density times the "
                             "active gravity-model magnitude times Load height.");
          prm.declare_entry ("Load density", "-1.0",
                             Patterns::Double(),
                             "Positive density contrast in kg/m^3 used with "
                             "Load height to compute surface-load traction.");
          prm.declare_entry ("Positive load direction", "unspecified",
                             Patterns::Selection("outward radial|inward radial|outward normal|inward normal|unspecified"),
                             "The physical direction of positive harmonic load. "
                             "For surface topographic load, if positive harmonic height "
                             "represents a mountain/load high, recommended setting is: "
                             "set Load magnitude = rho * g * h_amplitude and "
                             "set Positive load direction = inward radial.");

          // Legacy parameters for backward compatibility
          prm.declare_entry ("Amplitude", "1e300",
                             Patterns::Double(),
                             "Legacy parameter for scalar traction amplitude. Deprecated.");
          prm.declare_entry ("Sign", "1e300",
                             Patterns::Double(),
                             "Legacy parameter for traction sign multiplier. Deprecated.");
          prm.declare_entry ("Component", "unspecified",
                             Patterns::Selection("normal|radial|unspecified"),
                             "Legacy parameter for traction vector component. Deprecated.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    SphericalHarmonicLoad<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Spherical harmonic load");
        {
          harmonic_degree = prm.get_integer("Harmonic degree");
          harmonic_order = prm.get_integer("Harmonic order");
          coefficient_type = prm.get("Coefficient type");
          normalization = prm.get("Normalization");

          const double parsed_load_magnitude = prm.get_double("Load magnitude");
          const double parsed_load_height = prm.get_double("Load height");
          const double parsed_load_density = prm.get_double("Load density");
          const std::string parsed_direction = prm.get("Positive load direction");

          const double parsed_amplitude = prm.get_double("Amplitude");
          const double parsed_sign = prm.get_double("Sign");
          const std::string parsed_component = prm.get("Component");

          const bool has_height = (parsed_load_height != -1.0);
          const bool has_new = (parsed_load_magnitude != -1.0
                                || parsed_direction != "unspecified"
                                || has_height);
          const bool has_old = (parsed_amplitude != 1e300 || parsed_sign != 1e300 || parsed_component != "unspecified");

          use_load_height = false;
          load_height = 0.0;
          load_density = 0.0;

          if (has_height)
            {
              AssertThrow(!has_old,
                          ExcMessage("Do not combine Load height with the deprecated Amplitude, Sign, or Component parameters."));
              AssertThrow(parsed_load_magnitude == -1.0,
                          ExcMessage("Specify either Load height or Load magnitude, not both."));
              AssertThrow(parsed_load_height >= 0.0,
                          ExcMessage("Load height must be non-negative."));
              AssertThrow(parsed_load_density > 0.0,
                          ExcMessage("Load density must be positive when Load height is specified."));

              use_load_height = true;
              load_height = parsed_load_height;
              load_density = parsed_load_density;
              load_magnitude = 0.0;
              positive_load_direction =
                (parsed_direction == "unspecified" ? "outward normal" : parsed_direction);
            }
          else if (has_new && has_old)
            {
              double old_magnitude = std::abs((parsed_amplitude == 1e300 ? 0.0 : parsed_amplitude) * (parsed_sign == 1e300 ? 1.0 : parsed_sign));
              std::string old_comp = (parsed_component == "unspecified" ? "normal" : parsed_component);
              double product = (parsed_amplitude == 1e300 ? 0.0 : parsed_amplitude) * (parsed_sign == 1e300 ? 1.0 : parsed_sign);
              std::string old_direction;
              if (old_comp == "radial")
                old_direction = (product >= 0.0 ? "outward radial" : "inward radial");
              else
                old_direction = (product >= 0.0 ? "outward normal" : "inward normal");

              // Allow tiny tolerance for double comparisons
              if (std::abs(parsed_load_magnitude - old_magnitude) > 1e-5 || parsed_direction != old_direction)
                {
                  AssertThrow(false,
                              ExcMessage("You specified both the new parameter interface (Load magnitude, Positive load direction) "
                                         "and the old parameter interface (Amplitude, Sign, Component) for the Spherical harmonic load, "
                                         "and they are inconsistent. Please do not mix these interfaces."));
                }
              load_magnitude = parsed_load_magnitude;
              positive_load_direction = parsed_direction;
            }
          else if (has_new)
            {
              load_magnitude = parsed_load_magnitude;
              positive_load_direction = parsed_direction;
              if (load_magnitude == -1.0)
                load_magnitude = 0.0;
              if (positive_load_direction == "unspecified")
                positive_load_direction = "outward normal";
            }
          else if (has_old)
            {
              this->get_pcout() << "WARNING: You are using the deprecated parameters 'Amplitude', 'Sign', or 'Component' "
                                << "for the Spherical harmonic load boundary traction model. Please use "
                                << "'Load magnitude' and 'Positive load direction' instead." << std::endl;

              double amplitude_val = (parsed_amplitude == 1e300 ? 0.0 : parsed_amplitude);
              double sign_val = (parsed_sign == 1e300 ? 1.0 : parsed_sign);
              std::string comp_val = (parsed_component == "unspecified" ? "normal" : parsed_component);

              load_magnitude = std::abs(amplitude_val * sign_val);
              double product = amplitude_val * sign_val;
              if (comp_val == "radial")
                positive_load_direction = (product >= 0.0 ? "outward radial" : "inward radial");
              else
                positive_load_direction = (product >= 0.0 ? "outward normal" : "inward normal");
            }
          else
            {
              load_magnitude = 0.0;
              positive_load_direction = "outward normal";
            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(dim == 3,
                  ExcMessage("The spherical harmonic load boundary traction "
                             "plugin is only implemented for 3D spherical "
                             "geometries."));
      AssertThrow(harmonic_order <= harmonic_degree,
                  ExcMessage("Spherical harmonic order m must be smaller than "
                             "or equal to degree l."));
      if (harmonic_order == 0)
        AssertThrow(coefficient_type == "cosine",
                    ExcMessage("The sine coefficient is zero for m=0. Use "
                               "'Coefficient type = cosine'."));
      if (normalization == "unnormalized legendre")
        AssertThrow(harmonic_order == 0,
                    ExcMessage("The 'unnormalized legendre' normalization is "
                               "currently implemented only for order m=0. "
                               "Use 'geodesy 4pi' for m>0 loads."));

      AssertThrow(load_magnitude >= 0.0,
                  ExcMessage("Load magnitude must be non-negative. If you need to change the direction "
                             "of the load, please use the 'Positive load direction' parameter."));
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(SphericalHarmonicLoad,
                                            "spherical harmonic load",
                                            "Implementation of a boundary "
                                            "traction model that prescribes a "
                                            "single real spherical harmonic "
                                            "load by harmonic degree, harmonic order, "
                                            "normalization, load magnitude, "
                                            "and positive load direction. This is "
                                            "useful for benchmarks where the "
                                            "intended forcing mode is naturally "
                                            "written as an `(l,m)' spherical "
                                            "harmonic rather than as Cartesian "
                                            "traction components.")
  }
}
