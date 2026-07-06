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

#include <aspect/potential_feedback/interface.h>
#include <aspect/potential_feedback/tidal_potential.h>
#include <aspect/utilities.h>

#include <deal.II/base/patterns.h>
#include <algorithm>
#include <cctype>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace
    {
      std::string
      lowercase(const std::string &input)
      {
        std::string result = input;
        std::transform(result.begin(),
                       result.end(),
                       result.begin(),
                       [](const unsigned char c)
        {
          return static_cast<char>(std::tolower(c));
        });
        return result;
      }


      std::string
      normalize_interface_name(const std::string &name)
      {
        const std::string lower_name = lowercase(name);
        if (lower_name == "top" || lower_name == "outer"
            || lower_name == "surface")
          return "surface";
        if (lower_name == "bottom" || lower_name == "inner"
            || lower_name == "cmb")
          return "CMB";

        return name;
      }


      std::vector<std::string>
      normalize_interface_list(const std::string &input)
      {
        std::vector<std::string> result;
        for (const std::string &entry : Utilities::split_string_list(input))
          result.push_back(normalize_interface_name(entry));
        return result;
      }


      std::map<std::string, std::vector<std::string>>
      parse_selected_external_load_traction_indicators(
        const std::string &input)
      {
        std::map<std::string, std::vector<std::string>> result;
        if (Utilities::split_string_list(input).empty())
          return result;

        for (const std::string &entry : Utilities::split_string_list(input))
          {
            const std::vector<std::string> parts =
              Utilities::split_string_list(entry, ':');
            AssertThrow(parts.size() == 2,
                        ExcMessage("Potential feedback/Self gravity/"
                                   "Selected external load traction "
                                   "indicators entries must have the form "
                                   "`boundary: plugin name'. The entry <"
                                   + entry + "> does not match this form."));

            const std::string boundary =
              normalize_interface_name(parts[0]);
            std::vector<std::string> plugin_names =
              Utilities::split_string_list(parts[1], '|');
            if (plugin_names.empty())
              plugin_names.push_back(parts[1]);
            result[boundary].insert(result[boundary].end(),
                                    plugin_names.begin(),
                                    plugin_names.end());
          }

        return result;
      }


      DegreeOneReferenceFrame
      parse_degree_one_reference_frame(const std::string &name)
      {
        if (name == "none")
          return DegreeOneReferenceFrame::none;
        if (name == "geoid cancellation")
          return DegreeOneReferenceFrame::geoid_cancellation;
        if (name == "center of mass")
          return DegreeOneReferenceFrame::center_of_mass;
        if (name == "citcomsve center of mass")
          return DegreeOneReferenceFrame::citcomsve_center_of_mass;

        AssertThrow(false, ExcInternalError());
        return DegreeOneReferenceFrame::none;
      }
    }



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
                          Patterns::List(Patterns::Selection("self gravity|tidal potential|rotational feedback")),
                          "Comma-separated list of active potential-feedback "
                          "mechanisms. Supported names are `self gravity', "
                          "`tidal potential', and `rotational feedback'. "
                          "Mechanisms are activated by this list, not by "
                          "per-mechanism Enable flags.");

        prm.declare_entry("Interface source", "current interface displacement",
                          Patterns::Selection("current interface displacement|initial interface relief|prescribed load coefficients"),
                          "Source used to compute feedback potentials. The "
                          "default uses the current or predicted displacement "
                          "of density interfaces such as the surface and CMB.");

        prm.enter_subsection("Self gravity");
        {
          prm.declare_entry("Boundary indicators", "outer, inner",
                            Patterns::List(Patterns::Anything()),
                            "Boundary indicators that participate in "
                            "self-gravity potential feedback. The aliases "
                            "`outer', `top', and `surface' select the outer "
                            "surface; `inner', `bottom', and `CMB' select the "
                            "core-mantle boundary.");
          prm.declare_entry("External load source", "auto",
                            Patterns::Selection("none|auto|selected"),
                            "Which mechanical boundary tractions should be "
                            "converted to equivalent external-load height. "
                            "`none' ignores all traction loads; `auto' uses "
                            "only traction plugins that explicitly opt in as "
                            "potential-feedback load sources; `selected' uses "
                            "Selected external load traction indicators.");
          prm.declare_entry("Selected external load traction indicators", "",
                            Patterns::Anything(),
                            "Comma-separated map of boundary names to "
                            "boundary traction plugin names used when External "
                            "load source is `selected'. Use the form "
                            "`outer: spherical harmonic load'. Multiple plugin "
                            "names for one boundary can be separated by `|'.");
          prm.declare_entry("Minimum degree", "2",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree retained for "
                            "self-gravity potential feedback.");
          prm.declare_entry("Maximum degree", "32",
                            Patterns::Integer(0),
                            "Maximum spherical harmonic degree retained for "
                            "self-gravity potential feedback.");
          prm.declare_entry("Density above surface", "0.0",
                            Patterns::Double(0.0),
                            "density immediately above the surface in kg/m^3.");
          prm.declare_entry("Density below surface", "4604.4",
                            Patterns::Double(0.0),
                            "density immediately below the surface in kg/m^3.");
          prm.declare_entry("Density above CMB", "4604.4",
                            Patterns::Double(0.0),
                            "density immediately above the CMB in kg/m^3.");
          prm.declare_entry("Density below CMB", "10005.4",
                            Patterns::Double(0.0),
                            "density immediately below the CMB in kg/m^3.");
          prm.declare_entry("Include internal density anomalies", "auto",
                            Patterns::Selection("true|false|auto"),
                            "controls whether internal volume density anomalies contribute to "
                            "self-gravity/geoid diagnostics.");
          prm.declare_entry("Reference density for internal anomalies", "0.0",
                            Patterns::Double(0.0),
                            "A constant spherically symmetric reference density subtracted from the volume-density integral.");
          prm.declare_entry("Internal density anomaly tolerance", "0.0",
                            Patterns::Double(0.0),
                            "absolute tolerance for auto-detecting zero internal density anomaly field.");
          prm.declare_entry("Write self-gravity diagnostics", "false",
                            Patterns::Bool(),
                            "Whether to write self-gravity diagnostic text "
                            "output.");
          prm.declare_entry("Write coefficient diagnostics", "false",
                            Patterns::Bool(),
                            "Whether to write coefficient-level self-gravity "
                            "diagnostics.");
          prm.declare_entry("Time between diagnostic output", "0",
                            Patterns::Double(0),
                            "Time interval between self-gravity diagnostic "
                            "outputs. A value of zero disables time-based "
                            "diagnostic output.");
          prm.declare_entry("Time steps between diagnostic output", "0",
                            Patterns::Integer(0),
                            "Number of time steps between self-gravity "
                            "diagnostic outputs. A value of zero disables "
                            "step-based diagnostic output.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Tidal potential");
        {
          TidalPotential::declare_parameters(prm);
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

        prm.enter_subsection("Potential iteration");
        {
          prm.declare_entry("Relative tolerance", "1e-3",
                            Patterns::Double(0),
                            "Relative change tolerance for mechanism-specific "
                            "feedback-potential coefficient vectors.");
          prm.declare_entry("Maximum iterations", "20",
                            Patterns::Integer(1),
                            "Maximum number of self-consistent potential "
                            "updates per timestep. The iteration stops when "
                            "all active feedback-potential coefficient vectors "
                            "reach the relative tolerance or this limit is "
                            "reached.");
          prm.declare_entry("Freeze feedback after timestep zero", "false",
                            Patterns::Bool(),
                            "If true, retain the converged timestep-zero "
                            "feedback potential at later timesteps.");
          prm.declare_entry("Freeze after timestep zero", "false",
                            Patterns::Bool(),
                            "If true, retain the converged timestep-zero "
                            "feedback potential at later timesteps.");
          prm.declare_entry("Iterate with Stokes", "true",
                            Patterns::Bool(),
                            "Recompute feedback potentials from the current "
                            "Stokes velocity after every Stokes solve.");
          prm.declare_entry("Relaxation factor", "1.0",
                            Patterns::Double(0.0),
                            "Under-relaxation factor for potential iteration "
                            "coefficient updates.");
          prm.declare_entry("Initial displacement time step", "0",
                            Patterns::Double(0),
                            "Displacement interval used to convert the "
                            "timestep-0 Stokes velocity into an incremental "
                            "boundary displacement in seconds. If zero, "
                            "feedback mechanisms use the material model's "
                            "initial elastic time step when available.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Self-consistent potential update");
        {
          prm.declare_entry("Relative tolerance", "1e-3",
                            Patterns::Double(0),
                            "Deprecated compatibility alias for Potential "
                            "iteration/Relative tolerance.");
          prm.declare_entry("Maximum iterations", "20",
                            Patterns::Integer(1),
                            "Deprecated compatibility alias for Potential "
                            "iteration/Maximum iterations.");
          prm.declare_entry("Freeze feedback after timestep zero", "false",
                            Patterns::Bool(),
                            "Deprecated compatibility alias for Potential "
                            "iteration/Freeze after timestep zero.");
          prm.declare_entry("Iterate with Stokes", "true",
                            Patterns::Bool(),
                            "Deprecated compatibility alias for Potential "
                            "iteration/Iterate with Stokes.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Reference frame");
        {
          prm.declare_entry("Degree 1 reference frame",
                            "none",
                            Patterns::Selection("none|geoid cancellation|center of mass|citcomsve center of mass"),
                            "Degree-1 potential/reference-frame convention. "
                            "`none' leaves degree-1 potential unmodified. "
                            "`geoid cancellation' removes degree-1 potential "
                            "from the emitted boundary traction potential. "
                            "`center of mass' applies an ASPECT-native "
                            "reference-frame correction from the total "
                            "degree-1 mass dipole of active self-gravity "
                            "mass sources. `citcomsve center of mass' keeps "
                            "the benchmark-compatible CitcomSVE "
                            "incompressible degree-1 load-compensation "
                            "replay.");
          prm.declare_entry("Remove pure rotation from displacement", "true",
                            Patterns::Bool(),
                            "Whether to remove pure-rotation reference-frame "
                            "content from displacement diagnostics. This is "
                            "separate from ASPECT velocity nullspace removal.");
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

        prm.enter_subsection("Self gravity");
        {
          self_gravity_boundary_indicators =
            normalize_interface_list(prm.get("Boundary indicators"));
          self_gravity_source_interfaces = self_gravity_boundary_indicators;
          self_gravity_apply_boundaries = self_gravity_boundary_indicators;
          external_load_source = prm.get("External load source");
          selected_external_load_traction_indicators =
            parse_selected_external_load_traction_indicators(
              prm.get("Selected external load traction indicators"));
          self_gravity_min_degree = prm.get_integer("Minimum degree");
          self_gravity_max_degree = prm.get_integer("Maximum degree");
          self_gravity_density_above_surface = prm.get_double("Density above surface");
          self_gravity_density_below_surface = prm.get_double("Density below surface");
          self_gravity_density_above_cmb = prm.get_double("Density above CMB");
          self_gravity_density_below_cmb = prm.get_double("Density below CMB");
          include_internal_density_anomalies = prm.get("Include internal density anomalies");
          reference_density_for_internal_anomalies = prm.get_double("Reference density for internal anomalies");
          internal_density_anomaly_tolerance = prm.get_double("Internal density anomaly tolerance");
          write_self_gravity_diagnostics =
            prm.get_bool("Write self-gravity diagnostics");
          write_coefficient_diagnostics =
            prm.get_bool("Write coefficient diagnostics");
          time_between_diagnostic_output =
            prm.get_double("Time between diagnostic output");
          time_steps_between_diagnostic_output =
            prm.get_integer("Time steps between diagnostic output");
        }
        prm.leave_subsection();

        prm.enter_subsection("Tidal potential");
        {
          tidal_model_name = prm.get("Model name");
          prm.enter_subsection("Spherical harmonic potential");
          {
            tidal_harmonic_degree = prm.get_integer("Harmonic degree");
            tidal_harmonic_order = prm.get_integer("Harmonic order");
            tidal_coefficient_type = prm.get("Coefficient type");
            tidal_potential_quantity = prm.get("Potential quantity");
            tidal_potential_height_amplitude =
              prm.get_double("Potential height amplitude");
            tidal_potential_amplitude =
              prm.get_double("Potential amplitude");
            tidal_reference_gravity = prm.get_double("Reference gravity");
            tidal_normalization = prm.get("Normalization");
            tidal_time_dependence = prm.get("Time dependence");
            tidal_angular_frequency = prm.get_double("Angular frequency");
            tidal_phase = prm.get_double("Phase");
          }
          prm.leave_subsection();
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

        prm.enter_subsection("Potential iteration");
        {
          relative_tolerance = prm.get_double("Relative tolerance");
          maximum_iterations = prm.get_integer("Maximum iterations");
          freeze_feedback_after_timestep_zero =
            prm.get_bool("Freeze after timestep zero");
          iterate_with_stokes = prm.get_bool("Iterate with Stokes");
          potential_iteration_relaxation_factor =
            prm.get_double("Relaxation factor");
          initial_displacement_timestep =
            prm.get_double("Initial displacement time step");
        }
        prm.leave_subsection();

        prm.enter_subsection("Reference frame");
        {
          degree_one_reference_frame =
            parse_degree_one_reference_frame(prm.get("Degree 1 reference frame"));
          if (degree_one_reference_frame == DegreeOneReferenceFrame::none)
            {
              center_of_mass_correction = false;
              citcomsve_degree_one_load_compensation = false;
            }
          else if (degree_one_reference_frame ==
                   DegreeOneReferenceFrame::geoid_cancellation)
            {
              center_of_mass_correction = true;
              citcomsve_degree_one_load_compensation = false;
            }
          else if (degree_one_reference_frame ==
                   DegreeOneReferenceFrame::center_of_mass)
            {
              center_of_mass_correction = true;
              citcomsve_degree_one_load_compensation = false;
            }
          else if (degree_one_reference_frame ==
                   DegreeOneReferenceFrame::citcomsve_center_of_mass)
            {
              center_of_mass_correction = true;
              citcomsve_degree_one_load_compensation = true;
            }
          else
            AssertThrow(false, ExcInternalError());

          remove_pure_rotation_from_displacement =
            prm.get_bool("Remove pure rotation from displacement");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Populate interface_properties for compatibility with rotational/other feedback accesses
      interface_properties.surface.density_above = self_gravity_density_above_surface;
      interface_properties.surface.density_below = self_gravity_density_below_surface;
      interface_properties.cmb.density_above = self_gravity_density_above_cmb;
      interface_properties.cmb.density_below = self_gravity_density_below_cmb;

      AssertThrow(self_gravity_min_degree <= self_gravity_max_degree,
                  ExcMessage("Potential feedback/Self gravity/Minimum degree "
                             "must not exceed Maximum degree."));
      AssertThrow(rotational_min_degree <= rotational_max_degree,
                  ExcMessage("Potential feedback/Rotational feedback/Minimum "
                             "degree must not exceed Maximum degree."));
    }
  }
}
