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

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace
    {
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



      IceLoadReference
      parse_ice_load_reference(const std::string &name)
      {
        if (name == "first history file")
          return IceLoadReference::first_history_file;
        if (name == "zero thickness")
          return IceLoadReference::zero_thickness;

        AssertThrow(false, ExcInternalError());
        return IceLoadReference::first_history_file;
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
      prm.enter_subsection("Potential feedback");
      {
        prm.declare_entry("List of feedback mechanisms", "",
                          Patterns::List(Patterns::Selection("self gravity|tidal potential|rotational feedback|glacial isostatic adjustment")),
                          "Comma-separated list of active potential-feedback "
                          "mechanisms. Supported names are `self gravity', "
                          "`tidal potential', `rotational feedback', and "
                          "`glacial isostatic adjustment'. "
                          "Mechanisms are activated by this list, not by "
                          "per-mechanism Enable flags.");

        prm.enter_subsection("Self gravity");
        {
          prm.declare_entry("Maximum degree", "32",
                            Patterns::Integer(1),
                            "Maximum spherical harmonic degree retained for "
                            "self-gravity potential feedback. The calculation "
                            "starts internally at degree 1, matching "
                            "CitcomSVE's self-gravity potential synthesis.");
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
                            "self-gravity feedback and geoid diagnostics. "
                            "With 3-D mechanical mass conservation, the "
                            "resulting potential is also applied through the "
                            "full-domain compressible weak form.");
          prm.declare_entry("Reference density for internal anomalies", "0.0",
                            Patterns::Double(0.0),
                            "A constant spherically symmetric reference density subtracted from the volume-density integral.");
          prm.declare_entry("Internal density anomaly tolerance", "0.0",
                            Patterns::Double(0.0),
                            "absolute tolerance for auto-detecting zero internal density anomaly field.");
          prm.declare_entry("Full domain volume source discretization", "quadrature point",
                            Patterns::Selection("quadrature point|cell average|radial layer midpoint|mass lumped radial layer"),
                            "Discretization of the mechanical volume-density "
                            "source in the 3-D full-domain self-gravity "
                            "potential. `quadrature point' preserves the "
                            "existing pointwise integration. `cell average' "
                            "uses one volume-weighted density perturbation per "
                            "active cell before applying the spherical-harmonic "
                            "Green kernel. `radial layer midpoint' additionally "
                            "uses an arithmetic quadrature-point source average "
                            "and evaluates the radial kernel and radial measure "
                            "at the cell's midpoint radius. `mass lumped radial "
                            "layer' first projects those cell averages to "
                            "shared pressure vertices with a lumped Q1 mass "
                            "matrix before applying the midpoint rule. The "
                            "default is unchanged.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Tidal potential");
        {
          TidalPotential::declare_parameters(prm);
        }
        prm.leave_subsection();

        prm.enter_subsection("Rotational feedback");
        {
          prm.declare_entry("Fluid Love number", "1.0",
                            Patterns::Double(0),
                            "Fluid degree-2 Love number k_f used in the "
                            "linearized polar-wander relation. This is the "
                            "same quantity as CitcomSVE's polar_wander_kf. "
                            "Rotational feedback is internally the degree-2, "
                            "order-1 polar-wander forcing used by CitcomSVE.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Glacial isostatic adjustment");
        {
          prm.declare_entry("Ice load reference", "first history file",
                            Patterns::Selection("first history file|zero thickness"),
                            "Reference ice load. `first history file' applies "
                            "changes relative to the first stage; `zero "
                            "thickness' applies the absolute ice history.");
          prm.declare_entry("Ice density", "917.4",
                            Patterns::Double(0.0),
                            "Ice density in kg/m^3.");
          prm.declare_entry("Water density", "1000.0",
                            Patterns::Double(0.0),
                            "Ocean-water density in kg/m^3.");
          prm.declare_entry("Maximum degree", "32",
                            Patterns::Integer(1),
                            "Maximum spherical-harmonic degree retained for "
                            "ice, ocean, and total GIA surface loads.");
          prm.declare_entry("Output convergence diagnostics", "false",
                            Patterns::Bool(),
                            "Whether to print one compact GIA/SLE convergence "
                            "line after each counted post-Stokes surface-load "
                            "update. The diagnostic is disabled by default.");

          SurfaceHistory<3>::declare_parameters(
            prm,
            "Ice history",
            "$ASPECT_SOURCE_DIR/data/potential-feedback/gia/",
            "ice.%d.txt");
          SurfaceHistory<3>::declare_parameters(
            prm,
            "Prescribed ocean function history",
            "$ASPECT_SOURCE_DIR/data/potential-feedback/gia/",
            "ocean.%d.txt");
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
          prm.declare_alias("Freeze feedback after timestep zero",
                            "Freeze after timestep zero");
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
      prm.enter_subsection("Potential feedback");
      {
        feedback_mechanisms =
          Utilities::split_string_list(prm.get("List of feedback mechanisms"));

        prm.enter_subsection("Self gravity");
        {
          self_gravity_max_degree = prm.get_integer("Maximum degree");
          self_gravity_density_above_surface = prm.get_double("Density above surface");
          self_gravity_density_below_surface = prm.get_double("Density below surface");
          self_gravity_density_above_cmb = prm.get_double("Density above CMB");
          self_gravity_density_below_cmb = prm.get_double("Density below CMB");
          include_internal_density_anomalies = prm.get("Include internal density anomalies");
          reference_density_for_internal_anomalies = prm.get_double("Reference density for internal anomalies");
          internal_density_anomaly_tolerance = prm.get_double("Internal density anomaly tolerance");
          full_domain_volume_source_discretization =
            prm.get("Full domain volume source discretization");
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
          rotational_fluid_love_number = prm.get_double("Fluid Love number");
        }
        prm.leave_subsection();

        prm.enter_subsection("Glacial isostatic adjustment");
        {
          gia.ice_load_reference =
            parse_ice_load_reference(prm.get("Ice load reference"));
          gia.density_ice = prm.get_double("Ice density");
          gia.density_water = prm.get_double("Water density");
          gia.maximum_degree = prm.get_integer("Maximum degree");
          gia.output_convergence_diagnostics =
            prm.get_bool("Output convergence diagnostics");
          gia.ice_history =
            SurfaceHistory<3>::parse_parameters(prm, "Ice history");
          gia.ocean_history =
            SurfaceHistory<3>::parse_parameters(
              prm, "Prescribed ocean function history");
        }
        prm.leave_subsection();

        prm.enter_subsection("Potential iteration");
        {
          relative_tolerance = prm.get_double("Relative tolerance");
          maximum_iterations = prm.get_integer("Maximum iterations");
          freeze_feedback_after_timestep_zero =
            prm.get_bool("Freeze feedback after timestep zero");
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

      AssertThrow(self_gravity_max_degree >= 1,
                  ExcMessage("Potential feedback/Self gravity/Maximum degree "
                             "must be at least 1 because the self-gravity "
                             "calculation starts internally at degree 1."));

      const bool gia_is_active =
        std::find(feedback_mechanisms.begin(),
                  feedback_mechanisms.end(),
                  "glacial isostatic adjustment") != feedback_mechanisms.end();
      if (gia_is_active)
        {
          AssertThrow(std::find(feedback_mechanisms.begin(),
                                feedback_mechanisms.end(),
                                "self gravity") != feedback_mechanisms.end(),
                      ExcMessage("Glacial isostatic adjustment requires "
                                 "`self gravity' in Potential feedback/List "
                                 "of feedback mechanisms."));
          AssertThrow(gia.density_ice > 0.0 && gia.density_water > 0.0,
                      ExcMessage("GIA ice and water densities must be "
                                 "positive."));
        }
    }
  }
}
