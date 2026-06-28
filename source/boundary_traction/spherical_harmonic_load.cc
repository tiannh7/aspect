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

      if (component == "normal")
        return scalar * normal_vector;

      AssertThrow(component == "radial", ExcInternalError());
      Tensor<1,dim> radial_unit_vector;
      radial_unit_vector[0] = 1.;
      return scalar * Utilities::Coordinates::spherical_to_cartesian_vector(radial_unit_vector,
                                                                            position);
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
            Utilities::real_spherical_harmonic(degree, order, theta, phi);
          harmonic_value = (coefficient_type == "cosine" ? values.first : values.second);
        }
      else
        {
          AssertThrow(normalization == "unnormalized legendre", ExcInternalError());
          AssertThrow(order == 0,
                      ExcMessage("The 'unnormalized legendre' normalization is "
                                 "currently implemented only for order m=0. "
                                 "Use 'geodesy 4pi' for m>0 loads."));
          AssertThrow(coefficient_type == "cosine",
                      ExcMessage("The sine coefficient is zero for m=0. Use "
                                 "'Coefficient type = cosine'."));
          harmonic_value = legendre_p(degree, std::cos(theta));
        }

      return sign * amplitude * harmonic_value;
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
          prm.declare_entry ("Degree", "2",
                             Patterns::Integer(0),
                             "Spherical harmonic degree l.");
          prm.declare_entry ("Order", "0",
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
          prm.declare_entry ("Amplitude", "0",
                             Patterns::Double(),
                             "Scalar traction amplitude in Pa multiplying the "
                             "selected angular function.");
          prm.declare_entry ("Sign", "1",
                             Patterns::Double(),
                             "Diagnostic multiplier on the scalar load.");
          prm.declare_entry ("Component", "normal",
                             Patterns::Selection("normal|radial"),
                             "Direction of the traction vector. 'normal' "
                             "multiplies the scalar load by the outward face "
                             "normal supplied by ASPECT. 'radial' multiplies "
                             "it by the spherical radial unit vector.");
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
          degree = prm.get_integer("Degree");
          order = prm.get_integer("Order");
          coefficient_type = prm.get("Coefficient type");
          normalization = prm.get("Normalization");
          amplitude = prm.get_double("Amplitude");
          sign = prm.get_double("Sign");
          component = prm.get("Component");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(dim == 3,
                  ExcMessage("The spherical harmonic load boundary traction "
                             "plugin is only implemented for 3D spherical "
                             "geometries."));
      AssertThrow(order <= degree,
                  ExcMessage("Spherical harmonic order m must be smaller than "
                             "or equal to degree l."));
      if (order == 0)
        AssertThrow(coefficient_type == "cosine",
                    ExcMessage("The sine coefficient is zero for m=0. Use "
                               "'Coefficient type = cosine'."));
      if (normalization == "unnormalized legendre")
        AssertThrow(order == 0,
                    ExcMessage("The 'unnormalized legendre' normalization is "
                               "currently implemented only for order m=0. "
                               "Use 'geodesy 4pi' for m>0 loads."));
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
                                            "load by degree, order, amplitude, "
                                            "normalization, sign, and vector "
                                            "component. This is useful for "
                                            "benchmarks where the intended "
                                            "forcing mode is naturally written "
                                            "as an `(l,m)' spherical harmonic "
                                            "rather than as Cartesian traction "
                                            "components.")
  }
}
