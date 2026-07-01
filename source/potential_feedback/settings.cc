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
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

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
            prm.declare_entry("Density above", "-1e300",
                              Patterns::Double(),
                              "Deprecated.");
            prm.declare_entry("Density below", "-1e300",
                              Patterns::Double(),
                              "Deprecated.");
          }
          prm.leave_subsection();

          prm.enter_subsection("CMB");
          {
            prm.declare_entry("Density above", "-1e300",
                              Patterns::Double(),
                              "Deprecated.");
            prm.declare_entry("Density below", "-1e300",
                              Patterns::Double(),
                              "Deprecated.");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

        prm.enter_subsection("Self gravity");
        {
          prm.declare_entry("Source interfaces", "unspecified",
                            Patterns::Anything(),
                            "Deprecated.");
          prm.declare_entry("Apply to boundary indicators", "unspecified",
                            Patterns::Anything(),
                            "Deprecated.");
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
          prm.declare_entry("Maximum iterations", "10",
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
          prm.declare_entry("Iterate with Stokes", "true",
                            Patterns::Bool(),
                            "Recompute feedback potentials from the current "
                            "Stokes velocity after every Stokes solve.");
          prm.declare_entry("Initial displacement time step", "-1.e300",
                            Patterns::Double(),
                            "Deprecated.");
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
      }
      prm.leave_subsection();
    }



    void
    Settings::parse_parameters(ParameterHandler &prm)
    {
      dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      // Read global parameter before entering any subsection.
      const bool use_years = prm.get_bool("Use years instead of seconds");

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
          std::string apply_str = prm.get("Apply to boundary indicators");
          if (apply_str != "unspecified")
            {
              has_legacy_apply_boundaries = true;
              legacy_apply_boundaries = Utilities::split_string_list(apply_str);
            }
          self_gravity_min_degree = prm.get_integer("Minimum degree");
          self_gravity_max_degree = prm.get_integer("Maximum degree");
          self_gravity_density_above_surface = prm.get_double("Density above surface");
          self_gravity_density_below_surface = prm.get_double("Density below surface");
          self_gravity_density_above_cmb = prm.get_double("Density above CMB");
          self_gravity_density_below_cmb = prm.get_double("Density below CMB");
          include_internal_density_anomalies = prm.get("Include internal density anomalies");
          reference_density_for_internal_anomalies = prm.get_double("Reference density for internal anomalies");
          internal_density_anomaly_tolerance = prm.get_double("Internal density anomaly tolerance");
        }
        prm.leave_subsection();

        // Parse legacy Interface properties if specified
        double legacy_density_above_surface = -1e300;
        double legacy_density_below_surface = -1e300;
        double legacy_density_above_cmb = -1e300;
        double legacy_density_below_cmb = -1e300;

        prm.enter_subsection("Interface properties");
        {
          prm.enter_subsection("Surface");
          {
            legacy_density_above_surface = prm.get_double("Density above");
            legacy_density_below_surface = prm.get_double("Density below");
          }
          prm.leave_subsection();

          prm.enter_subsection("CMB");
          {
            legacy_density_above_cmb = prm.get_double("Density above");
            legacy_density_below_cmb = prm.get_double("Density below");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

        if (legacy_density_above_surface != -1e300 ||
            legacy_density_below_surface != -1e300 ||
            legacy_density_above_cmb != -1e300 ||
            legacy_density_below_cmb != -1e300)
          {
            pcout << "WARNING: Subsection 'Potential feedback / Interface properties' is deprecated. "
                              << "Please define interface densities directly in 'Potential feedback / Self gravity' instead." << std::endl;

            // Fill missing legacy values with defaults if some were not specified
            if (legacy_density_above_surface == -1e300) legacy_density_above_surface = 0.0;
            if (legacy_density_below_surface == -1e300) legacy_density_below_surface = 4604.4;
            if (legacy_density_above_cmb == -1e300) legacy_density_above_cmb = 4604.4;
            if (legacy_density_below_cmb == -1e300) legacy_density_below_cmb = 10005.4;

            // Consistency checks
            if (self_gravity_density_above_surface != 0.0 && self_gravity_density_above_surface != legacy_density_above_surface)
              AssertThrow(false, ExcMessage("Do not mix legacy 'Interface properties' and new 'Self gravity' density parameters."));
            if (self_gravity_density_below_surface != 4604.4 && self_gravity_density_below_surface != legacy_density_below_surface)
              AssertThrow(false, ExcMessage("Do not mix legacy 'Interface properties' and new 'Self gravity' density parameters."));
            if (self_gravity_density_above_cmb != 4604.4 && self_gravity_density_above_cmb != legacy_density_above_cmb)
              AssertThrow(false, ExcMessage("Do not mix legacy 'Interface properties' and new 'Self gravity' density parameters."));
            if (self_gravity_density_below_cmb != 10005.4 && self_gravity_density_below_cmb != legacy_density_below_cmb)
              AssertThrow(false, ExcMessage("Do not mix legacy 'Interface properties' and new 'Self gravity' density parameters."));

            self_gravity_density_above_surface = legacy_density_above_surface;
            self_gravity_density_below_surface = legacy_density_below_surface;
            self_gravity_density_above_cmb = legacy_density_above_cmb;
            self_gravity_density_below_cmb = legacy_density_below_cmb;
          }

        // Parse legacy Source interfaces if specified
        std::string legacy_source_interfaces = "unspecified";
        prm.enter_subsection("Self gravity");
        {
          legacy_source_interfaces = prm.get("Source interfaces");
        }
        prm.leave_subsection();

        if (legacy_source_interfaces != "unspecified" && legacy_source_interfaces != "surface, CMB")
          {
            pcout << "WARNING: Parameter 'Potential feedback / Self gravity / Source interfaces' is deprecated. "
                              << "Source interfaces are now inferred from explicit surface/CMB density parameters." << std::endl;
            self_gravity_source_interfaces = Utilities::split_string_list(legacy_source_interfaces);
          }
        else
          {
            self_gravity_source_interfaces = {"surface", "CMB"};
          }

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
          maximum_iterations = prm.get_integer("Maximum iterations");
          freeze_feedback_after_timestep_zero =
            prm.get_bool("Freeze feedback after timestep zero");
          iterate_with_stokes = prm.get_bool("Iterate with Stokes");
          initial_displacement_timestep =
            prm.get_double("Initial displacement time step");
          if (initial_displacement_timestep != -1e300)
            {
              has_legacy_initial_displacement_timestep = true;
              legacy_initial_displacement_timestep = initial_displacement_timestep;
              if (use_years)
                legacy_initial_displacement_timestep *= year_in_seconds;
            }
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
      }
      prm.leave_subsection();

      // Read legacy Postprocess / Geoid settings if specified
      std::string legacy_geoid_mode = "unspecified";
      double legacy_geoid_tolerance = -1e300;
      double legacy_geoid_ref_density = -1e300;
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Geoid");
        {
          legacy_geoid_mode = prm.get("Density anomaly contribution mode");
          legacy_geoid_tolerance = prm.get_double("Density anomaly tolerance");
          legacy_geoid_ref_density = prm.get_double("Reference density for anomaly");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      if (legacy_geoid_mode != "unspecified" || legacy_geoid_tolerance != -1e300 || legacy_geoid_ref_density != -1e300)
        {
          pcout << "WARNING: Parameters 'Postprocess / Geoid / Reference density for anomaly', "
                << "'Density anomaly contribution mode', and 'Density anomaly tolerance' are deprecated. "
                << "Please move them to 'Potential feedback / Self gravity' as "
                << "'Reference density for internal anomalies', 'Include internal density anomalies', "
                << "and 'Internal density anomaly tolerance'." << std::endl;

          std::string mapped_mode = "auto";
          if (legacy_geoid_mode == "always") mapped_mode = "true";
          else if (legacy_geoid_mode == "never") mapped_mode = "false";
          else if (legacy_geoid_mode == "auto") mapped_mode = "auto";
          else if (legacy_geoid_mode == "unspecified") mapped_mode = include_internal_density_anomalies;

          double mapped_tolerance = (legacy_geoid_tolerance == -1e300) ? internal_density_anomaly_tolerance : legacy_geoid_tolerance;
          double mapped_ref_density = (legacy_geoid_ref_density == -1e300) ? reference_density_for_internal_anomalies : legacy_geoid_ref_density;

          if (include_internal_density_anomalies != "auto" && include_internal_density_anomalies != mapped_mode)
            AssertThrow(false, ExcMessage("Conflict: legacy geoid and new self-gravity parameters are both set but inconsistent."));
          if (internal_density_anomaly_tolerance != 0.0 && internal_density_anomaly_tolerance != mapped_tolerance)
            AssertThrow(false, ExcMessage("Conflict: legacy geoid and new self-gravity parameters are both set but inconsistent."));
          if (reference_density_for_internal_anomalies != 0.0 && reference_density_for_internal_anomalies != mapped_ref_density)
            AssertThrow(false, ExcMessage("Conflict: legacy geoid and new self-gravity parameters are both set but inconsistent."));

          include_internal_density_anomalies = mapped_mode;
          internal_density_anomaly_tolerance = mapped_tolerance;
          reference_density_for_internal_anomalies = mapped_ref_density;
        }

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
