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

#include <aspect/potential_feedback/settings.h>
#include <aspect/utilities.h>

#include <deal.II/base/patterns.h>

namespace aspect
{
  namespace PotentialFeedback
  {
    double
    InterfaceDensity::density_contrast() const
    {
      return density_below - density_above;
    }



    bool
    Settings::has_active_mechanisms() const
    {
      return !feedback_mechanisms.empty();
    }



    void
    Settings::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Planet model");
      {
        prm.declare_entry("Model name", "earth benchmark",
                          Patterns::Anything(),
                          "Name of the planet model used by shared planetary "
                          "constants. The first potential-feedback migration "
                          "supports the `Custom' subsection for benchmark "
                          "constants.");

        prm.enter_subsection("Custom");
        {
          prm.declare_entry("Planet mass", "5.9722e24",
                            Patterns::Double(0),
                            "Planet mass in kg.");
          prm.declare_entry("Planet mean density", "5502.9914",
                            Patterns::Double(0),
                            "Mean density of the planet in kg/m^3.");
          prm.declare_entry("Polar moment of inertia", "8.034e37",
                            Patterns::Double(0),
                            "Polar moment of inertia in kg m^2.");
          prm.declare_entry("Equatorial moment of inertia", "8.010e37",
                            Patterns::Double(0),
                            "Equatorial moment of inertia in kg m^2.");
          prm.declare_entry("Rotation rate", "7.292115e-5",
                            Patterns::Double(0),
                            "Reference rotation rate in rad/s.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Potential feedback");
      {
        prm.declare_entry("List of feedback mechanisms", "",
                          Patterns::List(Patterns::Selection("self gravity|rotational feedback")),
                          "Comma-separated list of active potential-feedback "
                          "mechanisms. Supported names are `self gravity' and "
                          "`rotational feedback'. Mechanisms are activated by "
                          "this list, not by per-mechanism Enable flags.");

        prm.declare_entry("Interface source", "current interface displacement",
                          Patterns::Selection("current interface displacement|initial interface relief|prescribed load coefficients"),
                          "Source used to compute feedback potentials. The "
                          "default uses the current or predicted displacement "
                          "of density interfaces such as the surface and CMB.");

        prm.enter_subsection("Interface properties");
        {
          prm.enter_subsection("Surface");
          {
            prm.declare_entry("Density above", "0",
                              Patterns::Double(0),
                              "Density immediately above the surface in kg/m^3.");
            prm.declare_entry("Density below", "4604.4",
                              Patterns::Double(0),
                              "Density immediately below the surface in kg/m^3.");
          }
          prm.leave_subsection();

          prm.enter_subsection("CMB");
          {
            prm.declare_entry("Density above", "4604.4",
                              Patterns::Double(0),
                              "Density immediately above the CMB in kg/m^3.");
            prm.declare_entry("Density below", "10005.4",
                              Patterns::Double(0),
                              "Density immediately below the CMB in kg/m^3.");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

        prm.enter_subsection("Self gravity");
        {
          prm.declare_entry("Source interfaces", "surface, CMB",
                            Patterns::List(Patterns::Selection("surface|CMB")),
                            "Density interfaces that contribute to the "
                            "self-gravity potential computation.");
          prm.declare_entry("Apply to boundary indicators", "outer, inner",
                            Patterns::List(Patterns::Anything()),
                            "Boundary indicators that receive self-gravity "
                            "potential-feedback traction through the "
                            "`potential feedback traction' adapter.");
          prm.declare_entry("Minimum degree", "2",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree retained for "
                            "self-gravity potential feedback.");
          prm.declare_entry("Maximum degree", "32",
                            Patterns::Integer(0),
                            "Maximum spherical harmonic degree retained for "
                            "self-gravity potential feedback.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Rotational feedback");
        {
          prm.declare_entry("Inertia source interfaces", "surface",
                            Patterns::List(Patterns::Selection("surface|CMB")),
                            "Density interfaces used to compute inertia-tensor "
                            "perturbations for rotational feedback.");
          prm.declare_entry("Apply to boundary indicators", "outer",
                            Patterns::List(Patterns::Anything()),
                            "Boundary indicators that receive rotational "
                            "potential-feedback traction through the "
                            "`potential feedback traction' adapter.");
          prm.declare_entry("Minimum degree", "0",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree retained for "
                            "rotational-feedback diagnostics.");
          prm.declare_entry("Maximum degree", "2",
                            Patterns::Integer(0),
                            "Maximum spherical harmonic degree retained for "
                            "rotational-feedback diagnostics.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Self-consistent potential update");
        {
          prm.declare_entry("Relative tolerance", "1e-3",
                            Patterns::Double(0),
                            "Relative change tolerance for mechanism-specific "
                            "feedback-potential coefficient vectors.");
          prm.declare_entry("Freeze feedback after timestep zero", "false",
                            Patterns::Bool(),
                            "If true, retain the converged timestep-zero "
                            "feedback potential at later timesteps.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Reference frame");
        {
          prm.declare_entry("Center of mass correction", "true",
                            Patterns::Bool(),
                            "Whether to apply the degree-1 center-of-mass "
                            "reference-frame correction to potential-feedback "
                            "interface state.");
          prm.declare_entry("Remove pure rotation from displacement", "true",
                            Patterns::Bool(),
                            "Whether to remove pure-rotation reference-frame "
                            "content from displacement diagnostics. This is "
                            "separate from ASPECT velocity nullspace removal.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Diagnostics");
        {
          prm.declare_entry("Time between text output", "0.",
                            Patterns::Double(0.),
                            "Time interval between potential-feedback text "
                            "diagnostic outputs. A value of zero disables this "
                            "time-based limiter.");
          prm.declare_entry("Time steps between text output", "0",
                            Patterns::Integer(0),
                            "Time-step interval between potential-feedback "
                            "text diagnostic outputs. A value of zero disables "
                            "this step-based limiter.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    void
    Settings::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Planet model");
      {
        planet.model_name = prm.get("Model name");

        prm.enter_subsection("Custom");
        {
          planet.planet_mass = prm.get_double("Planet mass");
          planet.planet_mean_density = prm.get_double("Planet mean density");
          planet.polar_moment_of_inertia =
            prm.get_double("Polar moment of inertia");
          planet.equatorial_moment_of_inertia =
            prm.get_double("Equatorial moment of inertia");
          planet.rotation_rate = prm.get_double("Rotation rate");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Potential feedback");
      {
        feedback_mechanisms =
          Utilities::split_string_list(prm.get("List of feedback mechanisms"));
        interface_source = prm.get("Interface source");

        prm.enter_subsection("Interface properties");
        {
          prm.enter_subsection("Surface");
          {
            interface_properties.surface.density_above =
              prm.get_double("Density above");
            interface_properties.surface.density_below =
              prm.get_double("Density below");
          }
          prm.leave_subsection();

          prm.enter_subsection("CMB");
          {
            interface_properties.cmb.density_above =
              prm.get_double("Density above");
            interface_properties.cmb.density_below =
              prm.get_double("Density below");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

        prm.enter_subsection("Self gravity");
        {
          self_gravity_source_interfaces =
            Utilities::split_string_list(prm.get("Source interfaces"));
          self_gravity_apply_boundaries =
            Utilities::split_string_list(prm.get("Apply to boundary indicators"));
          self_gravity_min_degree = prm.get_integer("Minimum degree");
          self_gravity_max_degree = prm.get_integer("Maximum degree");
        }
        prm.leave_subsection();

        prm.enter_subsection("Rotational feedback");
        {
          rotational_inertia_source_interfaces =
            Utilities::split_string_list(prm.get("Inertia source interfaces"));
          rotational_apply_boundaries =
            Utilities::split_string_list(prm.get("Apply to boundary indicators"));
          rotational_min_degree = prm.get_integer("Minimum degree");
          rotational_max_degree = prm.get_integer("Maximum degree");
        }
        prm.leave_subsection();

        prm.enter_subsection("Self-consistent potential update");
        {
          relative_tolerance = prm.get_double("Relative tolerance");
          freeze_feedback_after_timestep_zero =
            prm.get_bool("Freeze feedback after timestep zero");
        }
        prm.leave_subsection();

        prm.enter_subsection("Reference frame");
        {
          center_of_mass_correction =
            prm.get_bool("Center of mass correction");
          remove_pure_rotation_from_displacement =
            prm.get_bool("Remove pure rotation from displacement");
        }
        prm.leave_subsection();

        prm.enter_subsection("Diagnostics");
        {
          time_between_text_output = prm.get_double("Time between text output");
          time_steps_between_text_output =
            prm.get_integer("Time steps between text output");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(self_gravity_min_degree <= self_gravity_max_degree,
                  ExcMessage("Potential feedback/Self gravity/Minimum degree "
                             "must not exceed Maximum degree."));
      AssertThrow(rotational_min_degree <= rotational_max_degree,
                  ExcMessage("Potential feedback/Rotational feedback/Minimum "
                             "degree must not exceed Maximum degree."));
    }
  }
}
