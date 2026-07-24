/*
  Copyright (C) 2024 by the authors of the ASPECT code.

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

#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/simulator.h>
#include <aspect/postprocess/boundary_densities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <iomanip>
#include <tuple>
#include <numeric>
#include <set>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace internal
    {
      double
      potential_traction_gravity(
        const bool has_full_domain_potential,
        const double full_domain_reference_gravity,
        const double local_gravity)
      {
        if (has_full_domain_potential)
          {
            AssertThrow(full_domain_reference_gravity > 0.0,
                        ExcMessage("A full-domain potential requires a "
                                   "positive reference gravity."));
            return full_domain_reference_gravity;
          }

        return local_gravity;
      }
    }

    namespace
    {
      bool
      self_gravity_list_contains(const std::vector<std::string> &values,
                                 const std::string &name)
      {
        return std::find(values.begin(), values.end(), name) != values.end();
      }


      bool
      print_self_gravity_diagnostic_once(
        const std::string &name,
        const unsigned int timestep_number,
        const unsigned int iteration_number)
      {
        static std::set<std::tuple<std::string, unsigned int, unsigned int>>
        printed_diagnostics;

        return printed_diagnostics
               .insert(std::make_tuple(name,
                                       timestep_number,
                                       iteration_number))
               .second;
      }

    }



    namespace internal
    {
      RadialGreenMomentAccumulator::RadialGreenMomentAccumulator(
        const std::vector<double> &evaluation_radii,
        const unsigned int minimum_degree,
        const unsigned int maximum_degree,
        const double reference_radius)
        : evaluation_radii(evaluation_radii)
        , reference_radius(reference_radius)
        , n_coefficients(0)
      {
        AssertThrow(!evaluation_radii.empty(),
                    ExcMessage("Radial Green moments require at least one evaluation radius."));
        AssertThrow(reference_radius > 0.0,
                    ExcMessage("The radial Green moment reference radius must be positive."));
        AssertThrow(maximum_degree >= minimum_degree,
                    ExcMessage("The radial Green moment degree range is invalid."));
        AssertThrow(std::is_sorted(evaluation_radii.begin(),
                                   evaluation_radii.end()),
                    ExcMessage("Radial Green moment evaluation radii must be sorted."));
        AssertThrow(std::adjacent_find(evaluation_radii.begin(),
                                       evaluation_radii.end())
                    == evaluation_radii.end(),
                    ExcMessage("Radial Green moment evaluation radii must be distinct."));
        AssertThrow(evaluation_radii.front() > 0.0,
                    ExcMessage("Radial Green moment evaluation radii must be positive."));

        for (unsigned int degree = minimum_degree;
             degree <= maximum_degree;
             ++degree)
          for (unsigned int order = 0; order <= degree; ++order)
            coefficient_degrees.push_back(degree);
        n_coefficients = coefficient_degrees.size();

        const unsigned int n_bins = evaluation_radii.size() + 1;
        inner_cos_moments.assign(n_bins * n_coefficients, 0.0);
        inner_sin_moments.assign(n_bins * n_coefficients, 0.0);
        outer_cos_moments.assign(n_bins * n_coefficients, 0.0);
        outer_sin_moments.assign(n_bins * n_coefficients, 0.0);
      }



      void
      RadialGreenMomentAccumulator::add_source(
        const double source_radius,
        const std::vector<double> &source_cos_coefficients,
        const std::vector<double> &source_sin_coefficients)
      {
        AssertThrow(source_radius > 0.0,
                    ExcMessage("A radial Green moment source radius must be positive."));
        AssertDimension(source_cos_coefficients.size(), n_coefficients);
        AssertDimension(source_sin_coefficients.size(), n_coefficients);

        const unsigned int source_bin =
          std::lower_bound(evaluation_radii.begin(),
                           evaluation_radii.end(),
                           source_radius)
          - evaluation_radii.begin();
        const double normalized_source_radius =
          source_radius / reference_radius;

        unsigned int previous_degree = numbers::invalid_unsigned_int;
        double inner_factor = 0.0;
        double outer_factor = 0.0;
        for (unsigned int coefficient_index = 0;
             coefficient_index < n_coefficients;
             ++coefficient_index)
          {
            const unsigned int degree =
              coefficient_degrees[coefficient_index];
            if (degree != previous_degree)
              {
                inner_factor =
                  std::pow(normalized_source_radius, degree);
                outer_factor =
                  std::pow(normalized_source_radius,
                           -static_cast<double>(degree + 1));
                previous_degree = degree;
              }
            const unsigned int index =
              source_bin * n_coefficients + coefficient_index;

            inner_cos_moments[index] +=
              source_cos_coefficients[coefficient_index] * inner_factor;
            inner_sin_moments[index] +=
              source_sin_coefficients[coefficient_index] * inner_factor;
            outer_cos_moments[index] +=
              source_cos_coefficients[coefficient_index] * outer_factor;
            outer_sin_moments[index] +=
              source_sin_coefficients[coefficient_index] * outer_factor;
          }
      }



      void
      RadialGreenMomentAccumulator::mpi_sum(
        const MPI_Comm &mpi_communicator)
      {
        const auto sum_moments = [&mpi_communicator](std::vector<double> &moments)
        {
          std::vector<double> global_moments(moments.size(), 0.0);
          dealii::Utilities::MPI::sum(moments,
                                      mpi_communicator,
                                      global_moments);
          moments.swap(global_moments);
        };

        sum_moments(inner_cos_moments);
        sum_moments(inner_sin_moments);
        sum_moments(outer_cos_moments);
        sum_moments(outer_sin_moments);
      }



      std::pair<std::vector<double>, std::vector<double>>
      RadialGreenMomentAccumulator::evaluate() const
      {
        const unsigned int n_radii = evaluation_radii.size();
        const unsigned int n_bins = n_radii + 1;
        std::vector<double> values_cos(n_radii * n_coefficients, 0.0);
        std::vector<double> values_sin(n_radii * n_coefficients, 0.0);
        std::vector<double> accumulated_inner_cos(n_coefficients, 0.0);
        std::vector<double> accumulated_inner_sin(n_coefficients, 0.0);
        std::vector<double> outer_suffix_cos(n_radii * n_coefficients, 0.0);
        std::vector<double> outer_suffix_sin(n_radii * n_coefficients, 0.0);
        std::vector<double> accumulated_outer_cos(n_coefficients, 0.0);
        std::vector<double> accumulated_outer_sin(n_coefficients, 0.0);

        for (unsigned int coefficient_index = 0;
             coefficient_index < n_coefficients;
             ++coefficient_index)
          {
            const unsigned int outermost_bin_index =
              (n_bins - 1) * n_coefficients + coefficient_index;
            accumulated_outer_cos[coefficient_index] =
              outer_cos_moments[outermost_bin_index];
            accumulated_outer_sin[coefficient_index] =
              outer_sin_moments[outermost_bin_index];
          }

        for (unsigned int reverse_radius_index = n_radii;
             reverse_radius_index > 0;
             --reverse_radius_index)
          {
            const unsigned int radius_index = reverse_radius_index - 1;
            const unsigned int bin = radius_index;
            for (unsigned int coefficient_index = 0;
                 coefficient_index < n_coefficients;
                 ++coefficient_index)
              {
                const unsigned int index =
                  bin * n_coefficients + coefficient_index;
                outer_suffix_cos[index] =
                  accumulated_outer_cos[coefficient_index];
                outer_suffix_sin[index] =
                  accumulated_outer_sin[coefficient_index];
                accumulated_outer_cos[coefficient_index] +=
                  outer_cos_moments[index];
                accumulated_outer_sin[coefficient_index] +=
                  outer_sin_moments[index];
              }
          }

        for (unsigned int radius_index = 0;
             radius_index < n_radii;
             ++radius_index)
          {
            const double normalized_evaluation_radius =
              evaluation_radii[radius_index] / reference_radius;
            unsigned int previous_degree = numbers::invalid_unsigned_int;
            double inner_factor = 0.0;
            double outer_factor = 0.0;
            for (unsigned int coefficient_index = 0;
                 coefficient_index < n_coefficients;
                 ++coefficient_index)
              {
                const unsigned int moment_index =
                  radius_index * n_coefficients + coefficient_index;
                accumulated_inner_cos[coefficient_index] +=
                  inner_cos_moments[moment_index];
                accumulated_inner_sin[coefficient_index] +=
                  inner_sin_moments[moment_index];

                const unsigned int degree =
                  coefficient_degrees[coefficient_index];
                if (degree != previous_degree)
                  {
                    inner_factor =
                      std::pow(normalized_evaluation_radius,
                               -static_cast<double>(degree + 1));
                    outer_factor =
                      std::pow(normalized_evaluation_radius, degree);
                    previous_degree = degree;
                  }
                const unsigned int value_index =
                  radius_index * n_coefficients + coefficient_index;
                values_cos[value_index] =
                  (accumulated_inner_cos[coefficient_index] * inner_factor
                   + outer_suffix_cos[moment_index] * outer_factor)
                  / reference_radius;
                values_sin[value_index] =
                  (accumulated_inner_sin[coefficient_index] * inner_factor
                   + outer_suffix_sin[moment_index] * outer_factor)
                  / reference_radius;
              }
          }

        return std::make_pair(values_cos, values_sin);
      }
    }



    template <int dim>
    void
    SelfGravitation<dim>::initialize()
    {
      top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      update_derived_planetary_constants();

      if (degree_one_reference_frame ==
          DegreeOneReferenceFrame::center_of_mass)
        {
          AssertThrow(false,
                      ExcMessage(
                        "The native coupled center-of-mass reference frame is "
                        "temporarily disabled. The current implementation does "
                        "not yet define a self-consistent reference-mass "
                        "contract: its COM translation was normalized by "
                        "g(R) R^2 / G even when the reference density, CMB "
                        "density, and gravity model are configured "
                        "independently. It also needs the COM dipole and "
                        "full-domain potential to use the same internal-source "
                        "time layer in every potential iteration. Use "
                        "`Degree 1 reference frame = none' or the legacy "
                        "`citcomsve center of mass' mode until the reference "
                        "mass and source-layer consistency fixes are added."));
          const auto &parameters = this->get_parameters();
          AssertThrow(parameters.density_source_law
                      == Parameters<dim>::Formulation::DensitySourceLaw::
                      mechanical_mass_conservation,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "the mechanical mass-conservation density source "
                        "law, so that its constraint and the Stokes forcing "
                        "use the same discrete mass source."));
          AssertThrow(self_gravity_mass_feedback_enabled,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "self gravity in Potential feedback/List of feedback "
                        "mechanisms."));
          AssertThrow(include_surface_contribution
                      && include_cmb_contribution,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "both surface and CMB mass feedback."));
          AssertThrow(include_internal_density_anomalies != "false",
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "internal density anomalies so that volume mass and "
                        "displaced internal interfaces are included in the "
                        "COM constraint."));
          AssertThrow(iterate_with_stokes,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "Potential feedback/Potential iteration/Iterate with "
                        "Stokes = true."));
          AssertThrow(!freeze_potential_after_timestep_zero,
                      ExcMessage(
                        "The coupled center-of-mass reference frame cannot "
                        "freeze its potential after timestep zero because the "
                        "mass-dipole constraint must follow the evolving "
                        "deformation."));
          AssertThrow(potential_iteration_relaxation_factor > 0.0
                      && potential_iteration_relaxation_factor <= 1.0,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "a potential-iteration relaxation factor in (0,1]."));
          AssertThrow(!(parameters.nullspace_removal
                        & Parameters<dim>::NullspaceRemoval::postsolve_translation),
                      ExcMessage(
                        "Do not combine the coupled center-of-mass reference "
                        "frame with net-translation or linear-momentum "
                        "nullspace removal because those options project the "
                        "velocity after the solve."));
          AssertThrow(parameters.nullspace_removal
                      & Parameters<dim>::NullspaceRemoval::angular_momentum,
                      ExcMessage(
                        "The coupled center-of-mass reference frame requires "
                        "Nullspace removal/Remove nullspace to include "
                        "'angular momentum'. A spherical free shell retains "
                        "a rigid-rotation nullspace, and this density-weighted "
                        "operation removes it without changing the mass "
                        "sources or COM constraint."));
          AssertThrow(!(parameters.nullspace_removal
                        & (Parameters<dim>::NullspaceRemoval::net_rotation
                           | Parameters<dim>::NullspaceRemoval::net_surface_rotation)),
                      ExcMessage(
                        "Use only the density-weighted 'angular momentum' "
                        "rotational nullspace operation with the coupled "
                        "center-of-mass reference frame."));
        }

      const double mm_initial_elastic_dt =
        this->get_material_model().initial_elastic_time_step();
      if (mm_initial_elastic_dt > 0.0 && initial_displacement_timestep == 0.0)
        initial_displacement_timestep = mm_initial_elastic_dt;

      if (configured_from_potential_feedback)
        {
          enable_surface_potential_traction = include_surface_contribution;
          enable_cmb_potential_traction = include_cmb_contribution;
        }

      last_text_output_time = -1.0;
      last_text_output_step = 0;
      current_tracked_step = (unsigned int)-1;
      printing_this_step = true;

      if (dim == 3)
        sh_transform = std::make_unique<Utilities::SphericalHarmonicTransform>(
                         max_degree, min_degree);
      else
        fourier_transform = std::make_unique<Utilities::FourierTransform>(
                              max_degree, min_degree);

      if (iterate_with_stokes)
        this->get_signals().post_stokes_solver.connect(
          [this](const SimulatorAccess<dim> &,
                 const unsigned int,
                 const unsigned int,
                 const SolverControl &,
                 const SolverControl &)
        {
          this->update_after_stokes_solve();
        });

      this->get_signals().post_mesh_deformation.connect(
        [this](const SimulatorAccess<dim> &)
      {
        this->compute_self_gravity_correction(false);
      });
    }



    template <int dim>
    void
    SelfGravitation<dim>::update_derived_planetary_constants()
    {
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Self-gravitation requires a spherical shell geometry."));

      const GeometryModel::SphericalShell<dim> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          this->get_geometry_model());

      const double outer_radius = geometry.outer_radius();
      const double surface_gravity =
        this->get_gravity_model()
        .gravity_vector(geometry.representative_point(0.0)).norm();

      AssertThrow(outer_radius > 0.0,
                  ExcMessage("Self-gravitation requires a positive outer "
                             "radius to derive planetary constants."));
      AssertThrow(surface_gravity > 0.0,
                  ExcMessage("Self-gravitation requires a positive surface "
                             "gravity magnitude to derive planetary constants."));

      planet_mass =
        surface_gravity * outer_radius * outer_radius / constants::big_g;
      planet_mean_density =
        3.0 * planet_mass
        / (4.0 * numbers::PI
           * outer_radius * outer_radius * outer_radius);

      AssertThrow(planet_mass > 0.0,
                  ExcMessage("Derived planet mass must be positive."));
      AssertThrow(planet_mean_density > 0.0,
                  ExcMessage("Derived planet mean density must be positive."));
    }



    template <int dim>
    std::vector<std::pair<std::vector<double>, std::vector<double>>>
    SelfGravitation<dim>::timed_spherical_harmonic_analysis_multiple(
      const std::vector<double> &theta,
      const std::vector<double> &phi,
      const std::vector<double> &weights,
      const std::vector<std::vector<double>> &values,
      const MPI_Comm &mpi_comm,
      const AnalysisBoundary analysis_boundary) const
    {
      const unsigned int n_fields = values.size();

      Utilities::SphericalHarmonicBasisCache &cache =
        (analysis_boundary == AnalysisBoundary::surface
         ? surface_analysis_basis_cache
         : cmb_analysis_basis_cache);

      if (n_fields == 0)
        return sh_transform->analyze_multiple(theta,
                                              phi,
                                              weights,
                                              values,
                                              mpi_comm);

      const Utilities::SphericalHarmonicBasisTable &basis =
        sh_transform->get_or_build_basis(theta,
                                         phi,
                                         cache);

      return sh_transform->analyze_multiple_with_basis(basis,
                                                       weights,
                                                       values,
                                                       mpi_comm);
    }



    template <int dim>
    std::vector<double>
    SelfGravitation<dim>::timed_spherical_harmonic_synthesis(
      const std::vector<double> &cos_coeffs,
      const std::vector<double> &sin_coeffs,
      const std::vector<double> &theta,
      const std::vector<double> &phi) const
    {
      return sh_transform->synthesize(cos_coeffs, sin_coeffs, theta, phi);
    }



    template <int dim>
    void
    SelfGravitation<dim>::update()
    {
      compute_self_gravity_correction(false);
    }


    template <int dim>
    void
    SelfGravitation<dim>::update_after_stokes_solve()
    {
      compute_self_gravity_correction(true);
    }


    template <int dim>
    void
    SelfGravitation<dim>::compute_self_gravity_correction(
      const bool include_current_velocity_increment)
    {
      if (freeze_potential_after_timestep_zero &&
          this->get_timestep_number() > 0)
        return;

      TimerOutput::Scope update_timer(this->get_computing_timer(),
                                      "Potential feedback: self-gravity update");

      const std::vector<double> old_surface_potential_cos =
        surface_potential_cos_coeffs;
      const std::vector<double> old_surface_potential_sin =
        surface_potential_sin_coeffs;
      const std::vector<double> old_cmb_potential_cos =
        cmb_potential_cos_coeffs;
      const std::vector<double> old_cmb_potential_sin =
        cmb_potential_sin_coeffs;
      const NativeCenterOfMassDiagnostic old_center_of_mass_diagnostic =
        native_center_of_mass_diagnostic;
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Self-gravitation requires a spherical shell geometry."));

      const GeometryModel::SphericalShell<dim> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          this->get_geometry_model());

      const double outer_radius = geometry.outer_radius();
      const double inner_radius = geometry.inner_radius();
      const double radius_ratio = inner_radius / outer_radius;
      double displacement_timestep = this->get_timestep();
      if (displacement_timestep == 0.0)
        displacement_timestep = initial_displacement_timestep;

      if (include_current_velocity_increment)
        {
          const unsigned int step = this->get_timestep_number();
          if (current_potential_iteration_step != step)
            {
              current_potential_iteration_step = step;
              potential_iteration_number = 0;
            }
          ++potential_iteration_number;
        }

      // Step 1: Collect surface and CMB topography at quadrature points
      // The projected field contains the Q2 Stokes velocity predictor. Using
      // the temperature degree (Q1 in this benchmark) supplies only one
      // quadrature point per face direction and biases the l=2 surface/CMB
      // coefficients by several percent. Integrate at least one order above
      // the velocity polynomial degree.
      const unsigned int quadrature_degree =
        std::max(2u,
                 this->introspection().polynomial_degree.velocities + 1u);
      const QGauss<dim - 1> quadrature_formula_face(quadrature_degree);

      FEFaceValues<dim> fe_face_values(this->get_mapping(),
                                       this->get_fe(),
                                       quadrature_formula_face,
                                       update_values |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

      const auto &mesh_deformation_handler =
        this->get_mesh_deformation_handler();
      const DoFHandler<dim> &mesh_deformation_dof_handler =
        mesh_deformation_handler.get_mesh_deformation_dof_handler();
      FEFaceValues<dim> mesh_face_values(
        this->get_mapping(),
        mesh_deformation_dof_handler.get_fe(),
        quadrature_formula_face,
        update_values);
      const FEValuesExtractors::Vector mesh_velocity_extractor(0);

      const LinearAlgebra::Vector *projected_mesh_velocity = nullptr;
      if (include_current_velocity_increment)
        projected_mesh_velocity =
          &mesh_deformation_handler.get_projected_free_surface_velocity(true);

      std::vector<Tensor<1,dim>> projected_mesh_velocity_values(
        mesh_face_values.n_quadrature_points);

      const double delta_rho_surf = density_below_surface - density_above_surface;
      const double delta_rho_cmb = density_below_cmb - density_above_cmb;

      const auto &traction_manager =
        this->get_boundary_traction_manager();
      const auto &plugins = traction_manager.get_active_plugins();
      const auto &plugin_boundaries =
        traction_manager.get_active_plugin_boundary_indicators();

      const auto load_traction_on_boundary =
        [&plugins,
         &plugin_boundaries,
         this]
        (const types::boundary_id boundary_id,
         const Point<dim> &position,
         const Tensor<1,dim> &face_normal)
      {
        Tensor<1,dim> load_traction;

        unsigned int plugin_index = 0;
        for (const auto &plugin : plugins)
          {
            const bool plugin_on_boundary =
              plugin_boundaries[plugin_index] == boundary_id;
            const bool is_feedback_plugin =
              dynamic_cast<const SelfGravitation<dim> *>(plugin.get()) != nullptr
              ||
              dynamic_cast<const PotentialFeedback::BoundaryTractionMarker *>(plugin.get()) != nullptr;

            bool use_plugin = false;
            if (plugin_on_boundary && !is_feedback_plugin)
              use_plugin = (configured_from_potential_feedback
                            || plugin->is_potential_feedback_load_source());

            if (use_plugin)
              load_traction += plugin->boundary_traction(boundary_id,
                                                         position,
                                                         face_normal);

            ++plugin_index;
          }

        if (additional_load_traction_function)
          load_traction += additional_load_traction_function(boundary_id,
                                                             position,
                                                             face_normal);

        return load_traction;
      };

      // Surface topography data
      std::vector<double> phi_pts;
      std::vector<double> theta_pts; // only used in 3D
      std::vector<double> weight_pts;
      std::vector<double> topo_pts;
      std::vector<double> surface_deformation_topo_pts;
      std::vector<double> external_load_topo_pts;

      // CMB topography data
      std::vector<double> cmb_phi_pts;
      std::vector<double> cmb_theta_pts; // only used in 3D
      std::vector<double> cmb_weight_pts;
      std::vector<double> cmb_topo_pts;
      std::vector<double> cmb_deformation_topo_pts;
      std::vector<double> cmb_committed_topo_pts;
      std::vector<double> cmb_external_load_topo_pts;

      {
        TimerOutput::Scope sample_timer(this->get_computing_timer(),
                                        "Potential feedback: collect boundary samples");
        auto mesh_cell = mesh_deformation_dof_handler.begin_active();
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          {
            const auto current_mesh_cell = mesh_cell;
            ++mesh_cell;
            if (cell->is_locally_owned() && cell->at_boundary())
              {
                for (const unsigned int f : cell->face_indices())
                  {
                    if (!cell->at_boundary(f))
                      continue;

                    const types::boundary_id bid = cell->face(f)->boundary_id();
                    const bool is_top    = (bid == top_boundary_id)
                                           && include_surface_contribution;
                    const bool is_bottom = (bid == bottom_boundary_id) && include_cmb_contribution;

                    if (!is_top && !is_bottom)
                      continue;

                    fe_face_values.reinit(cell, f);

                    if (include_current_velocity_increment)
                      {
                        mesh_face_values.reinit(current_mesh_cell, f);
                        mesh_face_values[mesh_velocity_extractor]
                        .get_function_values(*projected_mesh_velocity,
                                             projected_mesh_velocity_values);
                      }

                    for (unsigned int q = 0;
                         q < fe_face_values.n_quadrature_points;
                         ++q)
                      {
                        const Point<dim> position =
                          fe_face_values.quadrature_point(q);

                        const std::array<double, dim> scoord =
                          aspect::Utilities::Coordinates::
                          cartesian_to_spherical_coordinates(position);

                        // scoord: 2D = {r, phi}, 3D = {r, phi, theta}
                        const double ph = scoord[1]; // longitude / azimuthal angle
                        const Tensor<1,dim> radial_unit = position / scoord[0];
                        const double predicted_radial_displacement =
                          (include_current_velocity_increment
                           ? displacement_timestep
                           * (projected_mesh_velocity_values[q] * radial_unit)
                           : 0.0);

                        if (is_top)
                          {
                            const double h_rock =
                              this->get_geometry_model()
                              .height_above_reference_surface(position)
                              + predicted_radial_displacement;

                            // Compute the external load's equivalent height.
                            const Tensor<1,dim> face_normal =
                              fe_face_values.normal_vector(q);
                            const Tensor<1,dim> load_traction =
                              load_traction_on_boundary(top_boundary_id,
                                                        position,
                                                        face_normal);

                            // Inward load traction (T·n < 0) → positive surface mass
                            // σ_load = -T_load·n / g,  h_load = σ_load / Δρ
                            const double g_magnitude =
                              this->get_gravity_model().gravity_vector(position).norm();

                            double h_load = 0.0;
                            if (g_magnitude > 0 && delta_rho_surf > 0)
                              h_load = -(load_traction * face_normal) /
                                       (delta_rho_surf * g_magnitude);

                            const double h_effective = h_rock + h_load;

                            const double ref_radius = outer_radius;
                            const double w =
                              fe_face_values.JxW(q) /
                              (dim == 3 ? ref_radius *ref_radius : ref_radius);

                            phi_pts.push_back(ph);
                            if (dim == 3)
                              theta_pts.push_back(scoord[2]);
                            weight_pts.push_back(w);
                            topo_pts.push_back(h_effective);
                            surface_deformation_topo_pts.push_back(h_rock);
                            external_load_topo_pts.push_back(h_load);
                          }
                        else // is_bottom
                          {
                            const double r = scoord[0];
                            const double committed_cmb_topography = r - inner_radius;
                            const double cmb_topography =
                              committed_cmb_topography + predicted_radial_displacement;

                            const Tensor<1,dim> face_normal =
                              fe_face_values.normal_vector(q);
                            const Tensor<1,dim> load_traction =
                              load_traction_on_boundary(bottom_boundary_id,
                                                        position,
                                                        face_normal);

                            const double g_magnitude =
                              this->get_gravity_model().gravity_vector(position).norm();

                            double h_cmb_load = 0.0;
                            if (g_magnitude > 0 && delta_rho_cmb > 0)
                              h_cmb_load = (load_traction * face_normal) /
                                           (delta_rho_cmb * g_magnitude);

                            const double cmb_effective_topography =
                              cmb_topography + h_cmb_load;
                            const double ref_radius = inner_radius;
                            const double w =
                              fe_face_values.JxW(q) /
                              (dim == 3 ? ref_radius *ref_radius : ref_radius);

                            cmb_phi_pts.push_back(ph);
                            if (dim == 3)
                              cmb_theta_pts.push_back(scoord[2]);
                            cmb_weight_pts.push_back(w);
                            cmb_topo_pts.push_back(cmb_effective_topography);
                            cmb_deformation_topo_pts.push_back(cmb_topography);
                            cmb_committed_topo_pts.push_back(
                              committed_cmb_topography);
                            cmb_external_load_topo_pts.push_back(h_cmb_load);
                          }
                      }
                  }
              }
          }

        Assert(mesh_cell == mesh_deformation_dof_handler.end(),
               ExcInternalError());
      }

      // Step 2 & 3: SH/Fourier analysis + self-gravity kernel
      //
      // 3D self-gravity ratio: Rsg(l) = 3*delta_rho / ((2l+1)*rho_mean)
      // 2D self-gravity ratio: Rsg(n) = 2*delta_rho / (n * rho_mean)  [n>=1]
      //   (For n=0, Rsg=0 since uniform mass shift does not change the potential gradient.)
      //
      // CMB scaling: 3D: (r_cmb/R)^(l+2),  2D: (r_cmb/R)^(n+1)

      if (dim == 3)
        {
          const std::vector<std::pair<std::vector<double>, std::vector<double>>>
          surface_coefficients =
            timed_spherical_harmonic_analysis_multiple(
              theta_pts,
              phi_pts,
              weight_pts,
          {
            topo_pts,
            surface_deformation_topo_pts,
            external_load_topo_pts
          },
          this->get_mpi_communicator(),
          AnalysisBoundary::surface);
          AssertDimension(surface_coefficients.size(), 3);

          auto [cos_topo, sin_topo] = surface_coefficients[0];
          const unsigned int n_coeff = sh_transform->n_coefficients();
          auto [cos_surface_deformation, sin_surface_deformation] =
            surface_coefficients[1];
          auto [cos_external_load, sin_external_load] =
            surface_coefficients[2];

          std::vector<double> cos_cmb(n_coeff, 0.0);
          std::vector<double> sin_cmb(n_coeff, 0.0);
          std::vector<double> cos_cmb_deformation(n_coeff, 0.0);
          std::vector<double> sin_cmb_deformation(n_coeff, 0.0);
          std::vector<double> cos_cmb_external_load(n_coeff, 0.0);
          std::vector<double> sin_cmb_external_load(n_coeff, 0.0);
          // analyze() performs MPI collectives, so every rank must call it.
          // Ranks without locally owned CMB faces contribute empty vectors,
          // which correctly produce a zero local contribution.
          if (include_cmb_contribution)
            {
              const std::vector<std::pair<std::vector<double>, std::vector<double>>>
              cmb_coefficients =
                timed_spherical_harmonic_analysis_multiple(
                  cmb_theta_pts,
                  cmb_phi_pts,
                  cmb_weight_pts,
              {
                cmb_topo_pts,
                cmb_deformation_topo_pts,
                cmb_committed_topo_pts,
                cmb_external_load_topo_pts
              },
              this->get_mpi_communicator(),
              AnalysisBoundary::cmb);
              AssertDimension(cmb_coefficients.size(), 4);

              std::tie(cos_cmb, sin_cmb) = cmb_coefficients[0];
              std::tie(cos_cmb_deformation, sin_cmb_deformation) =
                cmb_coefficients[1];
              std::tie(cmb_committed_topography_cos_coeffs,
                       cmb_committed_topography_sin_coeffs) =
                         cmb_coefficients[2];
              std::tie(cos_cmb_external_load, sin_cmb_external_load) =
                cmb_coefficients[3];
            }

          if (!self_gravity_mass_feedback_enabled)
            {
              std::fill(cos_topo.begin(), cos_topo.end(), 0.0);
              std::fill(sin_topo.begin(), sin_topo.end(), 0.0);
              std::fill(cos_surface_deformation.begin(),
                        cos_surface_deformation.end(),
                        0.0);
              std::fill(sin_surface_deformation.begin(),
                        sin_surface_deformation.end(),
                        0.0);
              std::fill(cos_external_load.begin(),
                        cos_external_load.end(),
                        0.0);
              std::fill(sin_external_load.begin(),
                        sin_external_load.end(),
                        0.0);
              std::fill(cos_cmb.begin(), cos_cmb.end(), 0.0);
              std::fill(sin_cmb.begin(), sin_cmb.end(), 0.0);
              std::fill(cos_cmb_deformation.begin(),
                        cos_cmb_deformation.end(),
                        0.0);
              std::fill(sin_cmb_deformation.begin(),
                        sin_cmb_deformation.end(),
                        0.0);
              std::fill(cos_cmb_external_load.begin(),
                        cos_cmb_external_load.end(),
                        0.0);
              std::fill(sin_cmb_external_load.begin(),
                        sin_cmb_external_load.end(),
                        0.0);
            }

          cmb_topography_cos_coeffs = cos_cmb;
          cmb_topography_sin_coeffs = sin_cmb;

          native_center_of_mass_diagnostic =
            NativeCenterOfMassDiagnostic();
          if (degree_one_reference_frame ==
              DegreeOneReferenceFrame::center_of_mass
              && min_degree <= 1 && max_degree >= 1)
            {
              native_center_of_mass_diagnostic.valid = true;
              native_center_of_mass_diagnostic.surface_interface_dipole =
                degree_one_mass_dipole_from_height_coefficients(
                  cos_surface_deformation,
                  sin_surface_deformation,
                  delta_rho_surf,
                  outer_radius);
              native_center_of_mass_diagnostic.cmb_interface_dipole =
                degree_one_mass_dipole_from_height_coefficients(
                  cos_cmb_deformation,
                  sin_cmb_deformation,
                  delta_rho_cmb,
                  inner_radius);
              native_center_of_mass_diagnostic.external_load_dipole =
                degree_one_mass_dipole_from_height_coefficients(
                  cos_external_load,
                  sin_external_load,
                  delta_rho_surf,
                  outer_radius)
                +
                degree_one_mass_dipole_from_height_coefficients(
                  cos_cmb_external_load,
                  sin_cmb_external_load,
                  delta_rho_cmb,
                  inner_radius);
              native_center_of_mass_diagnostic.internal_density_dipole =
                compute_internal_density_mass_dipole();
              native_center_of_mass_diagnostic.mass_dipole_pre =
                native_center_of_mass_diagnostic.surface_interface_dipole
                + native_center_of_mass_diagnostic.cmb_interface_dipole
                + native_center_of_mass_diagnostic.external_load_dipole
                + native_center_of_mass_diagnostic.internal_density_dipole;
              native_center_of_mass_diagnostic.total_mass =
                (planet_mass > 0.0
                 ? planet_mass
                 : 4.0 * numbers::PI / 3.0
                 * planet_mean_density
                 * outer_radius * outer_radius * outer_radius);
              native_center_of_mass_diagnostic.correctable_mass =
                native_center_of_mass_diagnostic.total_mass;
              Tensor<1,3> unrelaxed_translation;
              for (unsigned int d = 0; d < 3; ++d)
                unrelaxed_translation[d] =
                  native_center_of_mass_diagnostic.mass_dipole_pre[d]
                  / native_center_of_mass_diagnostic.total_mass;

              if (include_current_velocity_increment
                  && old_center_of_mass_diagnostic.valid)
                for (unsigned int d = 0; d < 3; ++d)
                  native_center_of_mass_diagnostic.translation[d] =
                    (1.0 - potential_iteration_relaxation_factor)
                    * old_center_of_mass_diagnostic.translation[d]
                    + potential_iteration_relaxation_factor
                    * unrelaxed_translation[d];
              else
                native_center_of_mass_diagnostic.translation =
                  unrelaxed_translation;
              for (unsigned int d = 0; d < 3; ++d)
                native_center_of_mass_diagnostic.mass_dipole_post[d] =
                  native_center_of_mass_diagnostic.mass_dipole_pre[d]
                  - native_center_of_mass_diagnostic.total_mass
                  * native_center_of_mass_diagnostic.translation[d];
            }

          const unsigned int step = this->get_timestep_number();
          const double time = this->get_time();

          if (current_tracked_step != step)
            {
              current_tracked_step = step;
              printing_this_step = false;

              const double eff_time_interval = time_between_text_output;
              const unsigned int eff_step_interval = time_steps_between_text_output;

              if (eff_step_interval > 0 || eff_time_interval > 0.0)
                {
                  if (step == 0 || time == 0.0)
                    printing_this_step = true;
                  else if (eff_step_interval > 0 && (step - last_text_output_step >= eff_step_interval))
                    printing_this_step = true;
                  else if (eff_time_interval > 0 && (time - last_text_output_time >= eff_time_interval))
                    printing_this_step = true;
                }

              if (printing_this_step)
                {
                  last_text_output_step = step;
                  last_text_output_time = time;
                }
            }

          if (printing_this_step
              &&
              print_self_gravity_diagnostic_once("coefficient norms",
                                                 step,
                                                 potential_iteration_number))
            {
              const auto coefficient_l2_norm =
                [](const std::vector<double> &cos_coeffs,
                   const std::vector<double> &sin_coeffs)
              {
                double norm_squared = 0.0;
                for (const double value : cos_coeffs)
                  norm_squared += value * value;
                for (const double value : sin_coeffs)
                  norm_squared += value * value;
                return std::sqrt(norm_squared);
              };

              this->get_pcout()
                  << "      Self-gravity effective boundary SH coefficient L2 norm [m]:"
                  << std::scientific << std::setprecision(6)
                  << " surface=" << coefficient_l2_norm(cos_topo, sin_topo)
                  << ", CMB=" << coefficient_l2_norm(cos_cmb, sin_cmb)
                  << std::defaultfloat << std::endl;
            }

          // Phi/g at the surface.
          std::vector<double> surface_to_surface(max_degree + 1, 0.0);
          std::vector<double> cmb_to_surface(max_degree + 1, 0.0);
          // Phi/g at the CMB.
          std::vector<double> surface_to_cmb(max_degree + 1, 0.0);
          std::vector<double> cmb_to_cmb(max_degree + 1, 0.0);

          for (unsigned int l = min_degree; l <= max_degree; ++l)
            {
              const double common =
                3.0 / ((2.0 * l + 1.0) * planet_mean_density);
              surface_to_surface[l] = common * delta_rho_surf;
              cmb_to_surface[l] = common * delta_rho_cmb
                                  * std::pow(radius_ratio,
                                             static_cast<int>(l) + 2);
              surface_to_cmb[l] = common * delta_rho_surf
                                  * std::pow(radius_ratio,
                                             static_cast<int>(l));
              cmb_to_cmb[l] = common * delta_rho_cmb * radius_ratio;
            }

          surface_potential_cos_coeffs = cos_topo;
          surface_potential_sin_coeffs = sin_topo;
          sh_transform->apply_degree_filter(surface_potential_cos_coeffs,
                                            surface_potential_sin_coeffs,
                                            surface_to_surface);
          surface_mass_potential_cos_coeffs = surface_potential_cos_coeffs;
          surface_mass_potential_sin_coeffs = surface_potential_sin_coeffs;
          external_load_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          external_load_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          surface_deformation_mass_potential_cos_coeffs =
            surface_mass_potential_cos_coeffs;
          surface_deformation_mass_potential_sin_coeffs =
            surface_mass_potential_sin_coeffs;

          external_load_surface_potential_cos_coeffs = cos_external_load;
          external_load_surface_potential_sin_coeffs = sin_external_load;
          sh_transform->apply_degree_filter(
            external_load_surface_potential_cos_coeffs,
            external_load_surface_potential_sin_coeffs,
            surface_to_surface);

          surface_deformation_mass_potential_cos_coeffs =
            cos_surface_deformation;
          surface_deformation_mass_potential_sin_coeffs =
            sin_surface_deformation;
          sh_transform->apply_degree_filter(
            surface_deformation_mass_potential_cos_coeffs,
            surface_deformation_mass_potential_sin_coeffs,
            surface_to_surface);

          std::vector<double> cmb_at_surface_cos = cos_cmb;
          std::vector<double> cmb_at_surface_sin = sin_cmb;
          sh_transform->apply_degree_filter(cmb_at_surface_cos,
                                            cmb_at_surface_sin,
                                            cmb_to_surface);
          cmb_mass_potential_cos_coeffs = cmb_at_surface_cos;
          cmb_mass_potential_sin_coeffs = cmb_at_surface_sin;

          cmb_potential_cos_coeffs = cos_topo;
          cmb_potential_sin_coeffs = sin_topo;
          sh_transform->apply_degree_filter(cmb_potential_cos_coeffs,
                                            cmb_potential_sin_coeffs,
                                            surface_to_cmb);
          std::vector<double> cmb_at_cmb_cos = cos_cmb;
          std::vector<double> cmb_at_cmb_sin = sin_cmb;
          sh_transform->apply_degree_filter(cmb_at_cmb_cos,
                                            cmb_at_cmb_sin,
                                            cmb_to_cmb);

          for (unsigned int i = 0; i < n_coeff; ++i)
            {
              surface_potential_cos_coeffs[i] += cmb_at_surface_cos[i];
              surface_potential_sin_coeffs[i] += cmb_at_surface_sin[i];
              cmb_potential_cos_coeffs[i] += cmb_at_cmb_cos[i];
              cmb_potential_sin_coeffs[i] += cmb_at_cmb_sin[i];
            }

          update_full_domain_potential(cos_topo,
                                       sin_topo,
                                       cos_cmb,
                                       sin_cmb,
                                       outer_radius,
                                       inner_radius,
                                       include_current_velocity_increment
                                       || potential_iteration_number > 0);
          if (has_full_domain_potential())
            {
              surface_potential_cos_coeffs =
                full_domain_potential_cos_coeffs.back();
              surface_potential_sin_coeffs =
                full_domain_potential_sin_coeffs.back();
              cmb_potential_cos_coeffs =
                full_domain_potential_cos_coeffs.front();
              cmb_potential_sin_coeffs =
                full_domain_potential_sin_coeffs.front();
            }

          tidal_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_cos_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_sin_coeffs.assign(n_coeff, 0.0);
          degree_one_load_replay_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          degree_one_load_replay_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          citcomsve_degree_one_load_replay_diagnostic =
            CitcomSVEDegreeOneLoadReplayDiagnostic();

          cm_displacement_increment = Tensor<1,dim>();
          deformation_cm_displacement_increment = Tensor<1,dim>();
          external_load_cm_displacement_increment = Tensor<1,dim>();
          surface_deformation_cm_displacement_increment = Tensor<1,dim>();
          cmb_deformation_cm_displacement_increment = Tensor<1,dim>();
          reference_frame_acceleration = Tensor<1,dim>();

          if (native_center_of_mass_diagnostic.valid)
            {
              const unsigned int idx10 = sh_transform->index(1, 0);
              const unsigned int idx11 = sh_transform->index(1, 1);
              const double y1_normalization =
                std::sqrt(3.0 / (4.0 * numbers::PI));
              const Tensor<1,3> &translation =
                native_center_of_mass_diagnostic.translation;

              reference_frame_surface_potential_cos_coeffs[idx10] =
                -translation[2] / y1_normalization;
              reference_frame_surface_potential_cos_coeffs[idx11] =
                translation[0] / y1_normalization;
              reference_frame_surface_potential_sin_coeffs[idx11] =
                translation[1] / y1_normalization;

              double cmb_reference_frame_scale = 1.0;
              if (has_full_domain_potential())
                {
                  const double cmb_gravity =
                    this->get_gravity_model()
                    .gravity_vector(
                      geometry.representative_point(
                        geometry.maximal_depth())).norm();
                  cmb_reference_frame_scale =
                    cmb_gravity / full_domain_reference_gravity;
                }
              reference_frame_cmb_potential_cos_coeffs[idx10] =
                cmb_reference_frame_scale
                * reference_frame_surface_potential_cos_coeffs[idx10];
              reference_frame_cmb_potential_cos_coeffs[idx11] =
                cmb_reference_frame_scale
                * reference_frame_surface_potential_cos_coeffs[idx11];
              reference_frame_cmb_potential_sin_coeffs[idx11] =
                cmb_reference_frame_scale
                * reference_frame_surface_potential_sin_coeffs[idx11];

              // The full-domain cache already contains the reference-frame
              // potential at every radius. Retain this boundary-only path for
              // legacy formulations that do not construct that cache.
              if (!has_full_domain_potential())
                for (unsigned int order = 0; order <= 1; ++order)
                  {
                    const unsigned int index = sh_transform->index(1, order);
                    surface_potential_cos_coeffs[index] +=
                      reference_frame_surface_potential_cos_coeffs[index];
                    surface_potential_sin_coeffs[index] +=
                      reference_frame_surface_potential_sin_coeffs[index];
                    cmb_potential_cos_coeffs[index] +=
                      reference_frame_cmb_potential_cos_coeffs[index];
                    cmb_potential_sin_coeffs[index] +=
                      reference_frame_cmb_potential_sin_coeffs[index];
                  }

              for (unsigned int d = 0; d < dim; ++d)
                {
                  cm_displacement_increment[d] =
                    native_center_of_mass_diagnostic.translation[d];
                  external_load_cm_displacement_increment[d] =
                    native_center_of_mass_diagnostic.external_load_dipole[d]
                    / native_center_of_mass_diagnostic.total_mass;
                  surface_deformation_cm_displacement_increment[d] =
                    native_center_of_mass_diagnostic.surface_interface_dipole[d]
                    / native_center_of_mass_diagnostic.total_mass;
                  cmb_deformation_cm_displacement_increment[d] =
                    native_center_of_mass_diagnostic.cmb_interface_dipole[d]
                    / native_center_of_mass_diagnostic.total_mass;
                  deformation_cm_displacement_increment[d] =
                    (native_center_of_mass_diagnostic.surface_interface_dipole[d]
                     + native_center_of_mass_diagnostic.cmb_interface_dipole[d]
                     + native_center_of_mass_diagnostic.internal_density_dipole[d])
                    / native_center_of_mass_diagnostic.total_mass;
                }

              const std::vector<double> theta = {numbers::PI / 2.0,
                                                 numbers::PI / 2.0,
                                                 0.0
                                                };
              const std::vector<double> phi = {0.0,
                                               numbers::PI / 2.0,
                                               0.0
                                              };
              const std::vector<double> height =
                timed_spherical_harmonic_synthesis(
                  reference_frame_surface_potential_cos_coeffs,
                  reference_frame_surface_potential_sin_coeffs,
                  theta,
                  phi);
              const double surface_gravity =
                this->get_gravity_model()
                .gravity_vector(geometry.representative_point(0.0)).norm();
              for (unsigned int d = 0; d < dim; ++d)
                reference_frame_acceleration[d] =
                  -surface_gravity * height[d] / outer_radius;

              if (print_self_gravity_diagnostic_once(
                    "coupled center of mass",
                    this->get_timestep_number(),
                    potential_iteration_number))
                {
                  const auto print_vector =
                    [this](const Tensor<1,3> &vector)
                  {
                    this->get_pcout()
                        << "(" << vector[0]
                        << "," << vector[1]
                        << "," << vector[2] << ")";
                  };

                  this->get_pcout()
                      << "      Coupled center-of-mass constraint: "
                      << std::scientific << std::setprecision(6)
                      << "dipole pre [kg m]=";
                  print_vector(native_center_of_mass_diagnostic.mass_dipole_pre);
                  this->get_pcout() << ", translation [m]=";
                  print_vector(native_center_of_mass_diagnostic.translation);
                  this->get_pcout() << ", dipole residual [kg m]=";
                  print_vector(native_center_of_mass_diagnostic.mass_dipole_post);
                  this->get_pcout()
                      << ", volume source="
                      << full_domain_volume_source_discretization
                      << std::defaultfloat << std::endl;
                }

              write_native_center_of_mass_diagnostic(
                include_current_velocity_increment);
            }

          tidal_potential.add_to_coefficients(
            *sh_transform,
            radius_ratio,
            this->get_time(),
            surface_potential_cos_coeffs,
            surface_potential_sin_coeffs,
            cmb_potential_cos_coeffs,
            cmb_potential_sin_coeffs,
            tidal_surface_potential_cos_coeffs,
            tidal_surface_potential_sin_coeffs,
            tidal_cmb_potential_cos_coeffs,
            tidal_cmb_potential_sin_coeffs);

          if ((degree_one_reference_frame ==
               DegreeOneReferenceFrame::geoid_cancellation
               || degree_one_reference_frame ==
               DegreeOneReferenceFrame::citcomsve_center_of_mass)
              && min_degree <= 1 && max_degree >= 1)
            {
              // With normalized real degree-1 harmonics, the center-of-mass
              // displacement associated with a Phi/g coefficient is Phi_1m/g
              // times sqrt(3/(4*pi)). This reproduces CitcomSVE's initial
              // l=1 load CM: a 6.37 m surface load gives about 2.604 m of
              // CM_z.
              const unsigned int idx10 = sh_transform->index(1, 0);
              const unsigned int idx11 = sh_transform->index(1, 1);
              const double y1_normalization =
                std::sqrt(3.0 / (4.0 * numbers::PI));

              const auto potential_to_cm =
                [idx10, idx11, y1_normalization]
                (const std::vector<double> &cos_coeffs,
                 const std::vector<double> &sin_coeffs)
              {
                Tensor<1,dim> result;
                result[0] = -cos_coeffs[idx11] * y1_normalization;
                result[1] = -sin_coeffs[idx11] * y1_normalization;
                result[2] =  cos_coeffs[idx10] * y1_normalization;
                return result;
              };

              cm_displacement_increment =
                potential_to_cm(surface_potential_cos_coeffs,
                                surface_potential_sin_coeffs);
              external_load_cm_displacement_increment =
                potential_to_cm(external_load_surface_potential_cos_coeffs,
                                external_load_surface_potential_sin_coeffs);
              surface_deformation_cm_displacement_increment =
                potential_to_cm(surface_deformation_mass_potential_cos_coeffs,
                                surface_deformation_mass_potential_sin_coeffs);
              cmb_deformation_cm_displacement_increment =
                potential_to_cm(cmb_mass_potential_cos_coeffs,
                                cmb_mass_potential_sin_coeffs);

              if (citcomsve_degree_one_load_compensation)
                {
                  const double compensation_scale = 1.0;

                  degree_one_load_compensation_cos_coeffs[idx10] =
                    -compensation_scale
                    * external_load_surface_potential_cos_coeffs[idx10];
                  degree_one_load_compensation_cos_coeffs[idx11] =
                    -compensation_scale
                    * external_load_surface_potential_cos_coeffs[idx11];
                  degree_one_load_compensation_sin_coeffs[idx11] =
                    -compensation_scale
                    * external_load_surface_potential_sin_coeffs[idx11];

                  for (unsigned int order = 0; order <= 1; ++order)
                    {
                      const unsigned int index = sh_transform->index(1, order);
                      degree_one_load_replay_cmb_potential_cos_coeffs[index] =
                        cmb_potential_cos_coeffs[index]
                        + (surface_to_cmb[1] + cmb_to_cmb[1])
                        * degree_one_load_compensation_cos_coeffs[index];
                      degree_one_load_replay_cmb_potential_sin_coeffs[index] =
                        cmb_potential_sin_coeffs[index]
                        + (surface_to_cmb[1] + cmb_to_cmb[1])
                        * degree_one_load_compensation_sin_coeffs[index];
                    }

                  citcomsve_degree_one_load_replay_diagnostic.valid = true;
                  citcomsve_degree_one_load_replay_diagnostic
                  .original_surface_load_height_10 = cos_external_load[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_external_10_over_g =
                    external_load_surface_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.citcomsve_cm_z =
                    external_load_surface_potential_cos_coeffs[idx10]
                    * y1_normalization;
                  citcomsve_degree_one_load_replay_diagnostic.citcomsve_h_comp_10 =
                    degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .corrected_surface_load_height_10 =
                    cos_external_load[idx10]
                    + degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .corrected_cmb_load_height_10 =
                    degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.surface_kernel_l1 =
                    surface_to_surface[1];
                  citcomsve_degree_one_load_replay_diagnostic.cmb_kernel_l1 =
                    cmb_to_surface[1];
                  citcomsve_degree_one_load_replay_diagnostic
                  .net_degree1_phi_over_g_after_load_compensation =
                    external_load_surface_potential_cos_coeffs[idx10]
                    + (surface_to_surface[1] + cmb_to_surface[1])
                    * degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_pre_cancellation_over_g_10 =
                    cmb_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.h_comp_10 =
                    degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .surface_deformation_topo_cos_10 =
                    cos_surface_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_deformation_topo_cos_10 =
                    cos_cmb_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic.surface_to_cmb_l1 =
                    surface_to_cmb[1];
                  citcomsve_degree_one_load_replay_diagnostic.cmb_to_cmb_l1 =
                    cmb_to_cmb[1];
                  citcomsve_degree_one_load_replay_diagnostic
                  .surface_deformation_to_cmb_phi_over_g_10 =
                    surface_to_cmb[1] * cos_surface_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_deformation_to_cmb_phi_over_g_10 =
                    cmb_to_cmb[1] * cos_cmb_deformation[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .original_surface_load_to_cmb_l1_times_height =
                    surface_to_cmb[1] * cos_external_load[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .surface_to_cmb_l1_times_h_comp =
                    surface_to_cmb[1]
                    * degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_to_cmb_l1_times_h_comp =
                    cmb_to_cmb[1]
                    * degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_deformation_pre_compensation_over_g_10 =
                    citcomsve_degree_one_load_replay_diagnostic
                    .surface_deformation_to_cmb_phi_over_g_10
                    + citcomsve_degree_one_load_replay_diagnostic
                    .cmb_deformation_to_cmb_phi_over_g_10;
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_initial_load_pair_replay_over_g_10 =
                    citcomsve_degree_one_load_replay_diagnostic
                    .original_surface_load_to_cmb_l1_times_height
                    + citcomsve_degree_one_load_replay_diagnostic
                    .surface_to_cmb_l1_times_h_comp
                    + citcomsve_degree_one_load_replay_diagnostic
                    .cmb_to_cmb_l1_times_h_comp;
                  citcomsve_degree_one_load_replay_diagnostic
                  .phi_cmb_replay_over_g_10 =
                    degree_one_load_replay_cmb_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_intermediate_compensation_rhs_10 =
                    -degree_one_load_compensation_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_potential_append_rhs_10 =
                    degree_one_load_replay_cmb_potential_cos_coeffs[idx10];
                  citcomsve_degree_one_load_replay_diagnostic
                  .cmb_final_rhs_10 =
                    citcomsve_degree_one_load_replay_diagnostic
                    .cmb_intermediate_compensation_rhs_10
                    + citcomsve_degree_one_load_replay_diagnostic
                    .cmb_potential_append_rhs_10;
                }

              std::vector<double> deformation_potential_cos =
                surface_deformation_mass_potential_cos_coeffs;
              std::vector<double> deformation_potential_sin =
                surface_deformation_mass_potential_sin_coeffs;
              for (unsigned int i = 0; i < n_coeff; ++i)
                {
                  deformation_potential_cos[i] += cmb_mass_potential_cos_coeffs[i];
                  deformation_potential_sin[i] += cmb_mass_potential_sin_coeffs[i];
                }
              deformation_cm_displacement_increment =
                potential_to_cm(deformation_potential_cos,
                                deformation_potential_sin);

              // Zero degree-1 from the potential coefficients used for
              // boundary traction (geoid/k cancellation).  This is the
              // existing center_of_mass_correction: it ensures k1 = -1.
              if (center_of_mass_correction)
                {
                  for (unsigned int order = 0; order <= 1; ++order)
                    {
                      const unsigned int index = sh_transform->index(1, order);

                      reference_frame_surface_potential_cos_coeffs[index] =
                        -surface_potential_cos_coeffs[index];
                      reference_frame_surface_potential_sin_coeffs[index] =
                        -surface_potential_sin_coeffs[index];
                      reference_frame_cmb_potential_cos_coeffs[index] =
                        -cmb_potential_cos_coeffs[index];
                      reference_frame_cmb_potential_sin_coeffs[index] =
                        -cmb_potential_sin_coeffs[index];

                      surface_potential_cos_coeffs[index] = 0.0;
                      surface_potential_sin_coeffs[index] = 0.0;
                      cmb_potential_cos_coeffs[index] = 0.0;
                      cmb_potential_sin_coeffs[index] = 0.0;
                    }
                }

              const std::vector<double> theta = {numbers::PI / 2.0,
                                                 numbers::PI / 2.0,
                                                 0.0
                                                };
              const std::vector<double> phi = {0.0,
                                               numbers::PI / 2.0,
                                               0.0
                                              };
              const std::vector<double> height =
                timed_spherical_harmonic_synthesis(
                  reference_frame_surface_potential_cos_coeffs,
                  reference_frame_surface_potential_sin_coeffs,
                  theta,
                  phi);
              const double surface_gravity =
                this->get_gravity_model()
                .gravity_vector(geometry.representative_point(0.0)).norm();
              for (unsigned int d = 0; d < dim; ++d)
                reference_frame_acceleration[d] =
                  -surface_gravity * height[d] / outer_radius;
            }
        }
      else
        {
          auto [cos_topo, sin_topo] = fourier_transform->analyze(
                                        phi_pts, weight_pts, topo_pts,
                                        this->get_mpi_communicator());
          const unsigned int n_coeff = fourier_transform->n_coefficients();

          std::vector<double> cos_cmb(n_coeff, 0.0);
          std::vector<double> sin_cmb(n_coeff, 0.0);
          // analyze() performs MPI collectives, so every rank must call it.
          // Ranks without locally owned CMB faces contribute empty vectors,
          // which correctly produce a zero local contribution.
          if (include_cmb_contribution)
            {
              std::tie(cos_cmb, sin_cmb) = fourier_transform->analyze(
                                             cmb_phi_pts, cmb_weight_pts, cmb_topo_pts,
                                             this->get_mpi_communicator());
              std::tie(cmb_committed_topography_cos_coeffs,
                       cmb_committed_topography_sin_coeffs) =
                         fourier_transform->analyze(
                           cmb_phi_pts, cmb_weight_pts, cmb_committed_topo_pts,
                           this->get_mpi_communicator());
            }

          cmb_topography_cos_coeffs = cos_cmb;
          cmb_topography_sin_coeffs = sin_cmb;

          std::vector<double> surface_to_surface(max_degree + 1, 0.0);
          std::vector<double> cmb_to_surface(max_degree + 1, 0.0);
          std::vector<double> surface_to_cmb(max_degree + 1, 0.0);
          std::vector<double> cmb_to_cmb(max_degree + 1, 0.0);
          for (unsigned int n = std::max(min_degree, 1u); n <= max_degree; ++n)
            {
              const double common =
                2.0 / (static_cast<double>(n) * planet_mean_density);
              surface_to_surface[n] = common * delta_rho_surf;
              cmb_to_surface[n] = common * delta_rho_cmb
                                  * std::pow(radius_ratio,
                                             static_cast<int>(n) + 1);
              surface_to_cmb[n] = common * delta_rho_surf
                                  * std::pow(radius_ratio,
                                             static_cast<int>(n));
              cmb_to_cmb[n] = common * delta_rho_cmb * radius_ratio;
            }

          surface_potential_cos_coeffs = cos_topo;
          surface_potential_sin_coeffs = sin_topo;
          fourier_transform->apply_degree_filter(surface_potential_cos_coeffs,
                                                 surface_potential_sin_coeffs,
                                                 surface_to_surface);
          surface_mass_potential_cos_coeffs = surface_potential_cos_coeffs;
          surface_mass_potential_sin_coeffs = surface_potential_sin_coeffs;
          std::vector<double> cmb_at_surface_cos = cos_cmb;
          std::vector<double> cmb_at_surface_sin = sin_cmb;
          fourier_transform->apply_degree_filter(cmb_at_surface_cos,
                                                 cmb_at_surface_sin,
                                                 cmb_to_surface);
          cmb_mass_potential_cos_coeffs = cmb_at_surface_cos;
          cmb_mass_potential_sin_coeffs = cmb_at_surface_sin;

          cmb_potential_cos_coeffs = cos_topo;
          cmb_potential_sin_coeffs = sin_topo;
          fourier_transform->apply_degree_filter(cmb_potential_cos_coeffs,
                                                 cmb_potential_sin_coeffs,
                                                 surface_to_cmb);
          std::vector<double> cmb_at_cmb_cos = cos_cmb;
          std::vector<double> cmb_at_cmb_sin = sin_cmb;
          fourier_transform->apply_degree_filter(cmb_at_cmb_cos,
                                                 cmb_at_cmb_sin,
                                                 cmb_to_cmb);

          for (unsigned int i = 0; i < n_coeff; ++i)
            {
              surface_potential_cos_coeffs[i] += cmb_at_surface_cos[i];
              surface_potential_sin_coeffs[i] += cmb_at_surface_sin[i];
              cmb_potential_cos_coeffs[i] += cmb_at_cmb_cos[i];
              cmb_potential_sin_coeffs[i] += cmb_at_cmb_sin[i];
            }

          tidal_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          tidal_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_surface_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_surface_potential_sin_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_cos_coeffs.assign(n_coeff, 0.0);
          reference_frame_cmb_potential_sin_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_cos_coeffs.assign(n_coeff, 0.0);
          degree_one_load_compensation_sin_coeffs.assign(n_coeff, 0.0);
          native_center_of_mass_diagnostic =
            NativeCenterOfMassDiagnostic();
          reference_frame_acceleration = Tensor<1,dim>();
          cm_displacement_increment = Tensor<1,dim>();
          deformation_cm_displacement_increment = Tensor<1,dim>();
          external_load_cm_displacement_increment = Tensor<1,dim>();
          surface_deformation_cm_displacement_increment = Tensor<1,dim>();
          cmb_deformation_cm_displacement_increment = Tensor<1,dim>();
        }

      if (include_current_velocity_increment &&
          !old_surface_potential_cos.empty())
        {
          double difference_squared = 0.0;
          double new_norm_squared = 0.0;
          const auto accumulate_change =
            [&difference_squared, &new_norm_squared](
              const std::vector<double> &old_values,
              const std::vector<double> &new_values)
          {
            AssertDimension(old_values.size(), new_values.size());
            for (unsigned int i=0; i<new_values.size(); ++i)
              {
                difference_squared +=
                  (new_values[i]-old_values[i]) *
                  (new_values[i]-old_values[i]);
                new_norm_squared += new_values[i] * new_values[i];
              }
          };

          accumulate_change(old_surface_potential_cos,
                            surface_potential_cos_coeffs);
          accumulate_change(old_surface_potential_sin,
                            surface_potential_sin_coeffs);
          accumulate_change(old_cmb_potential_cos,
                            cmb_potential_cos_coeffs);
          accumulate_change(old_cmb_potential_sin,
                            cmb_potential_sin_coeffs);

          potential_relative_change =
            std::sqrt(difference_squared) /
            std::max(std::sqrt(new_norm_squared),
                     std::numeric_limits<double>::min());

          if (native_center_of_mass_diagnostic.valid
              && old_center_of_mass_diagnostic.valid)
            {
              const Tensor<1,3> translation_change =
                native_center_of_mass_diagnostic.translation
                - old_center_of_mass_diagnostic.translation;
              const double translation_scale =
                std::max(
                  std::max(native_center_of_mass_diagnostic.translation.norm(),
                           old_center_of_mass_diagnostic.translation.norm()),
                  100.0 * std::numeric_limits<double>::epsilon()
                  * outer_radius);
              center_of_mass_relative_change =
                translation_change.norm() / translation_scale;
              center_of_mass_absolute_change = translation_change.norm();
            }
          else if (native_center_of_mass_diagnostic.valid)
            {
              center_of_mass_relative_change =
                std::numeric_limits<double>::infinity();
              center_of_mass_absolute_change =
                std::numeric_limits<double>::infinity();
            }
          else
            {
              center_of_mass_relative_change = 0.0;
              center_of_mass_absolute_change = 0.0;
            }

          if (print_self_gravity_diagnostic_once("relative change",
                                                 this->get_timestep_number(),
                                                 potential_iteration_number))
            {
              this->get_pcout()
                  << "      Self-gravity potential update: "
                  << "relative SH coefficient change="
                  << std::scientific << std::setprecision(6)
                  << potential_relative_change;
              if (native_center_of_mass_diagnostic.valid)
                this->get_pcout()
                    << ", relative COM coefficient change="
                    << center_of_mass_relative_change
                    << ", absolute COM coefficient change [m]="
                    << center_of_mass_absolute_change;
              this->get_pcout() << std::defaultfloat << std::endl;

              const bool center_of_mass_change_is_converged =
                !native_center_of_mass_diagnostic.valid
                || center_of_mass_relative_change
                <= potential_convergence_tolerance
                || (center_of_mass_absolute_tolerance > 0.0
                    && center_of_mass_absolute_change
                    <= center_of_mass_absolute_tolerance);
              if ((potential_relative_change > potential_convergence_tolerance
                   || !center_of_mass_change_is_converged)
                  && potential_iteration_number >= maximum_potential_iterations)
                this->get_pcout()
                    << "        status=maximum iterations reached" << std::endl;
            }
        }

    }


    template <int dim>
    bool
    SelfGravitation<dim>::potential_is_converged() const
    {
      if (degree_one_reference_frame ==
          DegreeOneReferenceFrame::center_of_mass)
        {
          const bool converged =
            potential_relative_change <= potential_convergence_tolerance
            && (center_of_mass_relative_change
                <= potential_convergence_tolerance
                || (center_of_mass_absolute_tolerance > 0.0
                    && center_of_mass_absolute_change
                    <= center_of_mass_absolute_tolerance));
          AssertThrow(converged
                      || potential_iteration_number
                      < maximum_potential_iterations,
                      ExcMessage(
                        "The coupled center-of-mass reference-frame solve "
                        "reached the maximum number of potential iterations "
                        "without satisfying the potential tolerance and "
                        "either the relative or enabled absolute COM "
                        "coefficient-change tolerance."));
          return converged;
        }

      return potential_relative_change <= potential_convergence_tolerance
             || potential_iteration_number >= maximum_potential_iterations;
    }


    template <int dim>
    double
    SelfGravitation<dim>::potential_relative_change_value() const
    {
      return potential_relative_change;
    }



    template <int dim>
    double
    SelfGravitation<dim>::center_of_mass_relative_change_value() const
    {
      return center_of_mass_relative_change;
    }


    template <int dim>
    unsigned int
    SelfGravitation<dim>::minimum_degree() const
    {
      return min_degree;
    }


    template <int dim>
    bool
    SelfGravitation<dim>::has_full_domain_potential() const
    {
      return dim == 3
             && full_domain_reference_gravity > 0.0
             && full_domain_potential_radii.size() >= 2
             && full_domain_potential_cos_coeffs.size()
             == full_domain_potential_radii.size()
             && full_domain_potential_sin_coeffs.size()
             == full_domain_potential_radii.size();
    }


    template <int dim>
    double
    SelfGravitation<dim>::full_domain_potential(
      const Point<dim> &position) const
    {
      if constexpr (dim != 3)
        return 0.0;
      else
        {
          if (!has_full_domain_potential())
            return 0.0;

          const double radius = position.norm();
          AssertThrow(radius > 0.0,
                      ExcMessage("Full-domain self-gravity potential is undefined at radius zero."));

          const auto upper =
            std::upper_bound(full_domain_potential_radii.begin(),
                             full_domain_potential_radii.end(),
                             radius);
          unsigned int lower_index = 0;
          unsigned int upper_index = 0;
          double upper_weight = 0.0;

          if (upper == full_domain_potential_radii.begin())
            lower_index = upper_index = 0;
          else if (upper == full_domain_potential_radii.end())
            lower_index = upper_index =
                            full_domain_potential_radii.size() - 1;
          else
            {
              upper_index =
                std::distance(full_domain_potential_radii.begin(), upper);
              lower_index = upper_index - 1;
              upper_weight =
                (radius - full_domain_potential_radii[lower_index])
                / (full_domain_potential_radii[upper_index]
                   - full_domain_potential_radii[lower_index]);
            }

          const std::array<double,3> spherical_coordinates =
            aspect::Utilities::Coordinates::
            cartesian_to_spherical_coordinates(position);
          double potential_height = 0.0;
          unsigned int coefficient_index = 0;
          for (unsigned int degree = min_degree;
               degree <= max_degree;
               ++degree)
            for (unsigned int order = 0;
                 order <= degree;
                 ++order, ++coefficient_index)
              {
                const std::pair<double,double> harmonics =
                  aspect::Utilities::real_spherical_harmonic(
                    degree,
                    order,
                    spherical_coordinates[2],
                    spherical_coordinates[1]);
                const double cosine_coefficient =
                  (1.0 - upper_weight)
                  * full_domain_potential_cos_coeffs[lower_index]
                  [coefficient_index]
                  + upper_weight
                  * full_domain_potential_cos_coeffs[upper_index]
                  [coefficient_index];
                const double sine_coefficient =
                  (1.0 - upper_weight)
                  * full_domain_potential_sin_coeffs[lower_index]
                  [coefficient_index]
                  + upper_weight
                  * full_domain_potential_sin_coeffs[upper_index]
                  [coefficient_index];
                potential_height +=
                  cosine_coefficient * harmonics.first
                  + sine_coefficient * harmonics.second;
              }

          return full_domain_reference_gravity * potential_height;
        }
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::surface_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || surface_mass_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {surface_mass_potential_cos_coeffs.at(index),
              surface_mass_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::external_load_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || external_load_surface_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {external_load_surface_potential_cos_coeffs.at(index),
              external_load_surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::surface_deformation_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || surface_deformation_mass_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {surface_deformation_mass_potential_cos_coeffs.at(index),
              surface_deformation_mass_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::cmb_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || cmb_mass_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {cmb_mass_potential_cos_coeffs.at(index),
              cmb_mass_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::total_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || surface_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {surface_potential_cos_coeffs.at(index),
              surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::tidal_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || tidal_surface_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {tidal_surface_potential_cos_coeffs.at(index),
              tidal_surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    std::pair<double,double>
    SelfGravitation<dim>::reference_frame_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      AssertThrow(dim == 3,
                  ExcMessage("Spherical-harmonic coefficient access is only "
                             "available in 3D."));
      if (degree < min_degree || degree > max_degree || order > degree
          || reference_frame_surface_potential_cos_coeffs.empty())
        return {0.0, 0.0};

      const unsigned int index = sh_transform->index(degree, order);
      return {reference_frame_surface_potential_cos_coeffs.at(index),
              reference_frame_surface_potential_sin_coeffs.at(index)
             };
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::reference_frame_body_force(
      const Point<dim> &position) const
    {
      (void)position;
      return reference_frame_acceleration;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_cm_displacement_increment() const
    {
      return cm_displacement_increment;
    }


    template <int dim>
    bool
    SelfGravitation<dim>::uses_coupled_center_of_mass_reference_frame() const
    {
      return degree_one_reference_frame ==
             DegreeOneReferenceFrame::center_of_mass;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_deformation_cm_displacement_increment() const
    {
      return deformation_cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_external_load_cm_displacement_increment() const
    {
      return external_load_cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_surface_deformation_cm_displacement_increment() const
    {
      return surface_deformation_cm_displacement_increment;
    }


    template <int dim>
    Tensor<1,dim>
    SelfGravitation<dim>::get_cmb_deformation_cm_displacement_increment() const
    {
      return cmb_deformation_cm_displacement_increment;
    }


    template <int dim>
    double
    SelfGravitation<dim>::surface_density_jump() const
    {
      return density_below_surface - density_above_surface;
    }


    template <int dim>
    double
    SelfGravitation<dim>::cmb_density_jump() const
    {
      return density_below_cmb - density_above_cmb;
    }


    template <int dim>
    bool
    SelfGravitation<dim>::has_citcomsve_degree_one_load_replay_diagnostic() const
    {
      return citcomsve_degree_one_load_replay_diagnostic.valid;
    }


    template <int dim>
    double
    SelfGravitation<dim>::citcomsve_degree_one_cmb_intermediate_compensation_rhs_10() const
    {
      return citcomsve_degree_one_load_replay_diagnostic
             .cmb_intermediate_compensation_rhs_10;
    }


    template <int dim>
    double
    SelfGravitation<dim>::citcomsve_degree_one_cmb_potential_append_rhs_10() const
    {
      return citcomsve_degree_one_load_replay_diagnostic
             .cmb_potential_append_rhs_10;
    }


    template <int dim>
    double
    SelfGravitation<dim>::citcomsve_degree_one_cmb_final_rhs_10() const
    {
      return citcomsve_degree_one_load_replay_diagnostic.cmb_final_rhs_10;
    }


    template <int dim>
    double
    SelfGravitation<dim>::potential_height(
      const types::boundary_id boundary_indicator,
      const Point<dim> &position) const
    {
      if (surface_potential_cos_coeffs.empty())
        return 0.0;

      const bool is_surface = boundary_indicator == top_boundary_id;
      const bool is_cmb = boundary_indicator == bottom_boundary_id;
      if (!is_surface && !is_cmb)
        return 0.0;

      const std::vector<double> &potential_cos =
        (is_surface ? surface_potential_cos_coeffs : cmb_potential_cos_coeffs);
      const std::vector<double> &potential_sin =
        (is_surface ? surface_potential_sin_coeffs : cmb_potential_sin_coeffs);
      const std::array<double,dim> spherical_coordinates =
        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

      if constexpr (dim == 3)
        return timed_spherical_harmonic_synthesis(
                 potential_cos,
                 potential_sin,
        {spherical_coordinates[2]},
      {spherical_coordinates[1]})[0];

      return fourier_transform->synthesize(potential_cos,
                                           potential_sin,
      {spherical_coordinates[1]})[0];
    }



    template <int dim>
    void
    SelfGravitation<dim>::set_additional_load_traction_function(
      const std::function<Tensor<1,dim>(const types::boundary_id,
                                        const Point<dim> &,
                                        const Tensor<1,dim> &)> &function)
    {
      additional_load_traction_function = function;
    }



    template <int dim>
    Tensor<1, dim>
    SelfGravitation<dim>::boundary_traction(
      const types::boundary_id boundary_indicator,
      const Point<dim> &position,
      const Tensor<1, dim> &normal_vector) const
    {
      // Convert position to spherical coordinates
      const std::array<double, dim> scoord =
        aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(
          position);
      const double ph = scoord[1]; // longitude / azimuthal angle

      if (surface_potential_cos_coeffs.empty())
        return Tensor<1, dim>();

      const bool is_surface = boundary_indicator == top_boundary_id;
      const bool is_cmb = boundary_indicator == bottom_boundary_id;
      if (!is_surface && !is_cmb)
        return Tensor<1, dim>();

      bool displacement_history_supplies_local_restoring_traction = false;
      if (this->get_parameters().mesh_deformation_enabled)
        {
          const auto &mesh_deformation_handler =
            this->get_mesh_deformation_handler();
          displacement_history_supplies_local_restoring_traction =
            (mesh_deformation_handler
             .use_displacement_history_in_free_surface_stabilization()
             && mesh_deformation_handler
             .get_boundary_indicators_requiring_stabilization()
             .count(boundary_indicator) > 0);
        }

      const std::vector<double> &potential_cos =
        (is_surface ? surface_potential_cos_coeffs : cmb_potential_cos_coeffs);
      const std::vector<double> &potential_sin =
        (is_surface ? surface_potential_sin_coeffs : cmb_potential_sin_coeffs);

      double potential_height = 0.0;
      double cmb_topography = 0.0;
      double degree_one_load_compensation_topography = 0.0;
      double degree_one_load_replay_cmb_potential_height = 0.0;
      if (dim == 3)
        {
          const double th = scoord[2]; // colatitude
          const std::vector<double> th_vec = {th};
          const std::vector<double> ph_vec = {ph};
          const std::vector<double> potential =
            timed_spherical_harmonic_synthesis(
              potential_cos,
              potential_sin,
              th_vec,
              ph_vec);
          potential_height = potential[0];

          if (is_cmb
              && include_cmb_contribution
              && !displacement_history_supplies_local_restoring_traction)
            cmb_topography = timed_spherical_harmonic_synthesis(
                               cmb_committed_topography_cos_coeffs,
                               cmb_committed_topography_sin_coeffs,
                               th_vec,
                               ph_vec)[0];

          if (citcomsve_degree_one_load_compensation
              && !degree_one_load_compensation_cos_coeffs.empty())
            degree_one_load_compensation_topography =
              timed_spherical_harmonic_synthesis(
                degree_one_load_compensation_cos_coeffs,
                degree_one_load_compensation_sin_coeffs,
                th_vec,
                ph_vec)[0];

          if (is_cmb
              && citcomsve_degree_one_load_compensation
              && !degree_one_load_replay_cmb_potential_cos_coeffs.empty())
            degree_one_load_replay_cmb_potential_height =
              timed_spherical_harmonic_synthesis(
                degree_one_load_replay_cmb_potential_cos_coeffs,
                degree_one_load_replay_cmb_potential_sin_coeffs,
                th_vec,
                ph_vec)[0];

        }
      else
        {
          const std::vector<double> ph_vec = {ph};
          const std::vector<double> potential =
            fourier_transform->synthesize(potential_cos,
                                          potential_sin,
                                          ph_vec);
          potential_height = potential[0];

          if (is_cmb
              && include_cmb_contribution
              && !displacement_history_supplies_local_restoring_traction)
            cmb_topography = fourier_transform->synthesize(
                               cmb_committed_topography_cos_coeffs,
                               cmb_committed_topography_sin_coeffs,
                               ph_vec)[0];
        }

      const Tensor<1, dim> gravity =
        this->get_gravity_model().gravity_vector(position);
      const double g_magnitude = gravity.norm();
      const double potential_gravity =
        internal::potential_traction_gravity(
          has_full_domain_potential(),
          full_domain_reference_gravity,
          g_magnitude);
      const double delta_rho_cmb = density_below_cmb - density_above_cmb;

      if (is_surface)
        {
          double committed_surface_topography = 0.0;
          if (this->get_timestep_number() > 0
              && !displacement_history_supplies_local_restoring_traction)
            committed_surface_topography =
              this->get_geometry_model().height_above_reference_surface(position);

          // CitcomSVE keeps the current displacement increment in the local
          // restoring matrix and carries committed topography as an RHS load.
          // If ASPECT's free-surface stabilization already assembles that
          // displacement-history load, do not add the same local traction a
          // second time here. The committed topography remains part of the
          // non-local potential calculation.
          return density_below_surface
                 * (-g_magnitude * committed_surface_topography
                    + (enable_surface_potential_traction
                       ? potential_gravity * potential_height
                       : 0.0)) * normal_vector
                 - density_below_surface * g_magnitude
                 * degree_one_load_compensation_topography
                 * normal_vector;
        }

      // Fluid-core CMB condition after subtracting the mantle hydrostatic
      // reference state: Delta rho * (g*h_b - Phi_b) n.
      return delta_rho_cmb
             * (g_magnitude
                * (cmb_topography
                   + degree_one_load_compensation_topography)
                - potential_gravity
                * ((enable_cmb_potential_traction
                    ? potential_height
                    : 0.0)
                   + degree_one_load_replay_cmb_potential_height))
             * normal_vector;
    }



    template <int dim>
    void
    SelfGravitation<dim>::configure_from_potential_feedback_settings(
      const PotentialFeedback::Settings &settings)
    {
      max_degree = settings.self_gravity_max_degree;
      min_degree = 1;
      density_above_surface =
        settings.interface_properties.surface.density_above;
      density_below_surface =
        settings.interface_properties.surface.density_below;
      density_above_cmb =
        settings.interface_properties.cmb.density_above;
      density_below_cmb =
        settings.interface_properties.cmb.density_below;
      planet_mean_density = 0.0;
      planet_mass = 0.0;
      include_surface_contribution =
        settings.include_surface_feedback;
      include_cmb_contribution =
        settings.include_cmb_feedback;
      self_gravity_mass_feedback_enabled =
        self_gravity_list_contains(settings.feedback_mechanisms,
                                   "self gravity");
      iterate_with_stokes = settings.iterate_with_stokes;
      freeze_potential_after_timestep_zero =
        settings.freeze_feedback_after_timestep_zero;
      initial_displacement_timestep =
        settings.initial_displacement_timestep;
      potential_convergence_tolerance = settings.relative_tolerance;
      potential_iteration_relaxation_factor =
        settings.potential_iteration_relaxation_factor;
      maximum_potential_iterations = settings.maximum_iterations;
      include_internal_density_anomalies = settings.include_internal_density_anomalies;
      reference_density_for_internal_anomalies = settings.reference_density_for_internal_anomalies;
      internal_density_anomaly_tolerance = settings.internal_density_anomaly_tolerance;
      full_domain_volume_source_discretization =
        settings.full_domain_volume_source_discretization;
      full_domain_potential_radial_subdivisions =
        settings.full_domain_potential_radial_subdivisions;
      degree_one_reference_frame = settings.degree_one_reference_frame;
      center_of_mass_absolute_tolerance =
        settings.center_of_mass_absolute_tolerance;
      center_of_mass_correction = settings.center_of_mass_correction;
      citcomsve_degree_one_load_compensation =
        settings.citcomsve_degree_one_load_compensation;
      tidal_potential.configure_from_settings(settings,
                                              min_degree,
                                              max_degree,
                                              dim);
      configured_from_potential_feedback = true;
      time_between_text_output = 0.0;
      time_steps_between_text_output = 0;
      potential_relative_change = std::numeric_limits<double>::infinity();
      center_of_mass_relative_change =
        std::numeric_limits<double>::infinity();
      center_of_mass_absolute_change =
        std::numeric_limits<double>::infinity();
      current_potential_iteration_step = (unsigned int)-1;
      potential_iteration_number = 0;

      AssertThrow(max_degree >= min_degree,
                  ExcMessage("Potential feedback/Self gravity/Maximum degree "
                             "must be at least 1."));
      AssertThrow(include_surface_contribution || include_cmb_contribution,
                  ExcMessage("The `potential feedback' boundary traction model "
                             "must be prescribed on at least the top/surface or "
                             "bottom/CMB boundary."));
      AssertThrow(!citcomsve_degree_one_load_compensation
                  || (min_degree <= 1 && max_degree >= 1),
                  ExcMessage("CitcomSVE degree 1 load compensation requires "
                             "degree 1 to be included in the self-gravity "
                             "spherical-harmonic range."));
      AssertThrow(degree_one_reference_frame !=
                  DegreeOneReferenceFrame::center_of_mass
                  || dim == 3,
                  ExcMessage("The ASPECT-native degree-1 center-of-mass "
                             "reference frame is currently implemented only "
                             "for 3D spherical shells."));
      AssertThrow(degree_one_reference_frame !=
                  DegreeOneReferenceFrame::center_of_mass
                  || (min_degree <= 1 && max_degree >= 1),
                  ExcMessage("The ASPECT-native degree-1 center-of-mass "
                             "reference frame requires degree 1 to be "
                             "included in the self-gravity spherical-harmonic "
                             "range."));
    }



    template <int dim>
    void
    SelfGravitation<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Self gravitation");
        {
          prm.declare_entry("Maximum degree", "40",
                            Patterns::Integer(0),
                            "Maximum spherical harmonic degree for the "
                            "self-gravitation calculation.");

          prm.declare_entry("Minimum degree", "0",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree for the "
                            "self-gravitation calculation.");

          prm.declare_entry("Density above surface", "0",
                            Patterns::Double(0),
                            "Density immediately above the deformed surface "
                            "boundary in kg/m^3. For a free surface in vacuum "
                            "or thin atmosphere, set to 0. For a seafloor "
                            "under ocean, set to water density (e.g., 1030).");

          prm.declare_entry("Density below surface", "3500",
                            Patterns::Double(0),
                            "Density immediately below the deformed surface "
                            "boundary in kg/m^3. For rock topography, use "
                            "crustal density (e.g., 3500). For an ice cap "
                            "sitting on rock, use ice density (e.g., 917).");

          prm.declare_entry("Density above CMB", "5500",
                            Patterns::Double(0),
                            "Density immediately above the CMB (lower mantle side) "
                            "in kg/m^3. Earth: ~5500, Mars: ~3800.");

          prm.declare_entry("Density below CMB", "9900",
                            Patterns::Double(0),
                            "Density immediately below the CMB (outer core side) "
                            "in kg/m^3. Earth: ~9900, Mars: ~6200.");

          prm.declare_entry("Include CMB contribution", "true",
                            Patterns::Bool(),
                            "Whether to include the CMB topography contribution "
                            "to the self-gravitational potential perturbation. "
                            "Set to false if only surface topography feedback is needed.");

          prm.declare_entry("Iterate with Stokes", "true",
                            Patterns::Bool(),
                            "Recompute the non-local surface/CMB potential from "
                            "the current Stokes velocity after every Stokes solve. "
                            "The updated traction is used by the next nonlinear "
                            "iteration in the same time step.");
          prm.declare_entry("Freeze potential after timestep zero", "false",
                            Patterns::Bool(),
                            "Diagnostic switch that retains the converged "
                            "timestep-zero non-local potential coefficients "
                            "without recomputing them at later timesteps.");

          prm.declare_entry("Initial displacement time step", "0",
                            Patterns::Double(0),
                            "Displacement interval used to convert the timestep-0 "
                            "Stokes velocity into an incremental boundary displacement. "
                            "Set this to the elastic time step for an instantaneously "
                            "applied load. Units are years when 'Use years instead of "
                            "seconds' is enabled, otherwise seconds.");
          prm.declare_entry("Potential convergence tolerance", "1e-3",
                            Patterns::Double(0),
                            "Relative L2 change tolerance for the combined "
                            "surface and CMB Phi/g spherical-harmonic "
                            "coefficient vectors. Zhong et al. (2022) author "
                            "inputfile10 uses 1e-3 for its self-gravity "
                            "iteration cutoff.");
          prm.declare_entry("Maximum potential iterations", "10",
                            Patterns::Integer(1),
                            "Maximum number of self-consistent potential "
                            "updates per timestep. The iteration stops when "
                            "the potential coefficient change reaches the "
                            "tolerance or this limit is reached.");
          prm.declare_entry("Enable surface potential traction", "true",
                            Patterns::Bool(),
                            "Diagnostic switch controlling whether Phi/g is "
                            "applied as a non-local traction at the outer "
                            "surface. Harmonic analysis and output remain "
                            "active when this switch is false.");
          prm.declare_entry("Enable CMB potential traction", "true",
                            Patterns::Bool(),
                            "Diagnostic switch controlling whether Phi/g is "
                            "applied as a non-local traction at the CMB. The "
                            "local CMB topography term is unaffected.");
          prm.declare_entry("Center of mass correction", "false",
                            Patterns::Bool(),
                            "Whether to apply the degree-1 center-of-mass "
                            "reference-frame correction. This correction only "
                            "affects degree 1 and is separate from ASPECT "
                            "nullspace removal.");
          prm.declare_entry("CitcomSVE degree 1 load compensation", "false",
                            Patterns::Bool(),
                            "Whether to apply the CitcomSVE-style degree-1 "
                            "center-of-mass compensating load before solving "
                            "the displacement response. This diagnostic option "
                            "is disabled by default and is separate from "
                            "ASPECT nullspace removal and from the degree-1 "
                            "geoid reference-frame correction.");
          prm.declare_entry("Include internal density anomalies", "auto",
                            Patterns::Selection("true|false|auto"),
                            "Whether to include the internal mantle density anomalies "
                            "contribution to the gravitational potential. Default is auto.");
          prm.declare_entry("Reference density for internal anomalies", "0",
                            Patterns::Double(),
                            "Reference density used to define mantle density anomalies (kg/m^3).");
          prm.declare_entry("Internal density anomaly tolerance", "0",
                            Patterns::Double(0),
                            "Density anomaly threshold below which the volume integral is skipped.");
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
          prm.declare_entry("Full domain potential radial subdivisions", "32",
                            Patterns::Integer(1),
                            "Number of uniform radial intervals used to cache "
                            "the 3-D full-domain self-gravity potential. "
                            "Tabulated reference-density radii are added to "
                            "these support points. Internal volume and sheet "
                            "sources are evaluated efficiently with cumulative "
                            "radial Green-function moments. A value of 1 "
                            "reproduces the former two-boundary-point cache "
                            "for a constant reference-density model.");
          prm.declare_entry("Time between text output", "0.",
                            Patterns::Double(0.),
                            "The time interval in years between text outputs for self-gravity diagnostics. "
                            "If zero, this parameter is ignored.");
          prm.declare_entry("Time steps between text output", "0",
                            Patterns::Integer(0),
                            "The number of time steps between self-gravity diagnostic text outputs. "
                            "If zero, this parameter is ignored. If both output interval parameters are zero, no self-gravity diagnostic text is printed.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    SelfGravitation<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Self gravitation");
        {
          max_degree = prm.get_integer("Maximum degree");
          min_degree = prm.get_integer("Minimum degree");
          density_above_surface = prm.get_double("Density above surface");
          density_below_surface = prm.get_double("Density below surface");
          density_above_cmb = prm.get_double("Density above CMB");
          density_below_cmb = prm.get_double("Density below CMB");
          planet_mean_density = 0.0;
          planet_mass = 0.0;
          include_cmb_contribution = prm.get_bool("Include CMB contribution");
          iterate_with_stokes = prm.get_bool("Iterate with Stokes");
          freeze_potential_after_timestep_zero =
            prm.get_bool("Freeze potential after timestep zero");
          initial_displacement_timestep =
            prm.get_double("Initial displacement time step");
          potential_convergence_tolerance =
            prm.get_double("Potential convergence tolerance");
          maximum_potential_iterations =
            prm.get_integer("Maximum potential iterations");
          enable_surface_potential_traction =
            prm.get_bool("Enable surface potential traction");
          enable_cmb_potential_traction =
            prm.get_bool("Enable CMB potential traction");
          center_of_mass_correction =
            prm.get_bool("Center of mass correction");
          citcomsve_degree_one_load_compensation =
            prm.get_bool("CitcomSVE degree 1 load compensation");
          if (citcomsve_degree_one_load_compensation)
            degree_one_reference_frame =
              DegreeOneReferenceFrame::citcomsve_center_of_mass;
          else if (center_of_mass_correction)
            degree_one_reference_frame =
              DegreeOneReferenceFrame::geoid_cancellation;
          else
            degree_one_reference_frame = DegreeOneReferenceFrame::none;
          include_surface_contribution = true;
          self_gravity_mass_feedback_enabled = true;
          potential_iteration_relaxation_factor = 1.0;
          include_internal_density_anomalies =
            prm.get("Include internal density anomalies");
          reference_density_for_internal_anomalies =
            prm.get_double("Reference density for internal anomalies");
          internal_density_anomaly_tolerance =
            prm.get_double("Internal density anomaly tolerance");
          full_domain_volume_source_discretization =
            prm.get("Full domain volume source discretization");
          full_domain_potential_radial_subdivisions =
            prm.get_integer("Full domain potential radial subdivisions");
          time_between_text_output = prm.get_double("Time between text output");
          time_steps_between_text_output = prm.get_integer("Time steps between text output");
          potential_relative_change = std::numeric_limits<double>::infinity();
          center_of_mass_relative_change =
            std::numeric_limits<double>::infinity();
          center_of_mass_absolute_change =
            std::numeric_limits<double>::infinity();
          current_potential_iteration_step = (unsigned int)-1;
          potential_iteration_number = 0;

          if (this->convert_output_to_years())
            {
              initial_displacement_timestep *= year_in_seconds;
              time_between_text_output *= year_in_seconds;
            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(min_degree <= max_degree,
                  ExcMessage("Minimum degree must not exceed Maximum degree."));
      AssertThrow(!citcomsve_degree_one_load_compensation
                  || (min_degree <= 1 && max_degree >= 1),
                  ExcMessage("Boundary traction model/Self gravitation/"
                             "CitcomSVE degree 1 load compensation requires "
                             "degree 1 to be included in the self-gravity "
                             "spherical-harmonic range."));
    }


    template <int dim>
    std::string
    SelfGravitation<dim>::get_include_internal_density_anomalies() const
    {
      return include_internal_density_anomalies;
    }

    template <int dim>
    double
    SelfGravitation<dim>::get_reference_density_for_internal_anomalies() const
    {
      return reference_density_for_internal_anomalies;
    }

    template <int dim>
    double
    SelfGravitation<dim>::get_internal_density_anomaly_tolerance() const
    {
      return internal_density_anomaly_tolerance;
    }

    template <int dim>
    std::pair<std::vector<double>, std::vector<double>>
    SelfGravitation<dim>::to_spherical_harmonic_coefficients(
      const std::vector<std::vector<double>> &spherical_function) const
    {
      std::vector<double> cosi(spherical_function.size(), 0.0);
      std::vector<double> sini(spherical_function.size(), 0.0);
      std::vector<double> coecos;
      std::vector<double> coesin;

      for (unsigned int ideg = min_degree; ideg < max_degree + 1; ++ideg)
        {
          for (unsigned int iord = 0; iord < ideg + 1; ++iord)
            {
              // Do the spherical harmonic expansion.
              for (unsigned int ds_num = 0; ds_num < spherical_function.size(); ++ds_num)
                {
                  // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                  const std::pair<double, double> sph_harm_vals =
                    aspect::Utilities::real_spherical_harmonic(ideg, iord, spherical_function.at(ds_num).at(0), spherical_function.at(ds_num).at(1));
                  const double cos_component = sph_harm_vals.first;
                  const double sin_component = sph_harm_vals.second;

                  cosi.at(ds_num) = (spherical_function.at(ds_num).at(3) * cos_component);
                  sini.at(ds_num) = (spherical_function.at(ds_num).at(3) * sin_component);
                }
              // Integrate the contribution of each spherical infinitesimal.
              double cosii = 0;
              double sinii = 0;
              for (unsigned int ds_num = 0; ds_num < spherical_function.size(); ++ds_num)
                {
                  cosii += cosi.at(ds_num) * spherical_function.at(ds_num).at(2);
                  sinii += sini.at(ds_num) * spherical_function.at(ds_num).at(2);
                }
              coecos.push_back(cosii);
              coesin.push_back(sinii);
            }
        }
      // Sum over each processor.
      dealii::Utilities::MPI::sum (coecos, this->get_mpi_communicator(), coecos);
      dealii::Utilities::MPI::sum (coesin, this->get_mpi_communicator(), coesin);

      return std::make_pair(coecos, coesin);
    }


    template <int dim>
    Tensor<1,3>
    SelfGravitation<dim>::degree_one_mass_dipole_from_height_coefficients(
      const std::vector<double> &cos_coeffs,
      const std::vector<double> &sin_coeffs,
      const double density_jump,
      const double radius) const
    {
      Tensor<1,3> dipole;
      if (dim != 3 || min_degree > 1 || max_degree < 1 || cos_coeffs.empty())
        return dipole;

      const unsigned int idx10 = sh_transform->index(1, 0);
      const unsigned int idx11 = sh_transform->index(1, 1);
      const double y1_normalization =
        std::sqrt(3.0 / (4.0 * numbers::PI));
      const double scale =
        density_jump * radius * radius * radius / y1_normalization;

      dipole[0] = -scale * cos_coeffs[idx11];
      dipole[1] = -scale * sin_coeffs[idx11];
      dipole[2] =  scale * cos_coeffs[idx10];
      return dipole;
    }


    template <int dim>
    Tensor<1,3>
    SelfGravitation<dim>::compute_internal_density_mass_dipole() const
    {
      AssertThrow(false, ExcNotImplemented());
      return Tensor<1,3>();
    }


    template <int dim>
    void
    SelfGravitation<dim>::update_full_domain_potential(
      const std::vector<double> &,
      const std::vector<double> &,
      const std::vector<double> &,
      const std::vector<double> &,
      const double,
      const double,
      const bool)
    {
      full_domain_potential_radii.clear();
      full_domain_potential_cos_coeffs.clear();
      full_domain_potential_sin_coeffs.clear();
      full_domain_reference_gravity = 0.0;
    }


    template <>
    void
    SelfGravitation<3>::update_full_domain_potential(
      const std::vector<double> &surface_height_cos,
      const std::vector<double> &surface_height_sin,
      const std::vector<double> &cmb_height_cos,
      const std::vector<double> &cmb_height_sin,
      const double outer_radius,
      const double inner_radius,
      const bool include_internal_sources)
    {
      full_domain_potential_radii.clear();
      full_domain_potential_cos_coeffs.clear();
      full_domain_potential_sin_coeffs.clear();
      full_domain_reference_gravity = 0.0;

      const auto &parameters = this->get_parameters();
      if (!self_gravity_mass_feedback_enabled
          || parameters.density_source_law
          != Parameters<3>::Formulation::DensitySourceLaw::mechanical_mass_conservation)
        return;

      for (unsigned int radial_index = 0;
           radial_index <= full_domain_potential_radial_subdivisions;
           ++radial_index)
        full_domain_potential_radii.push_back(
          (radial_index == 0
           ? inner_radius
           : (radial_index == full_domain_potential_radial_subdivisions
              ? outer_radius
              : inner_radius
              + (outer_radius - inner_radius)
              * (static_cast<double>(radial_index)
                 / full_domain_potential_radial_subdivisions))));
      if (parameters.reference_density_model
          == Parameters<3>::Formulation::ReferenceDensityModel::tabulated_radial)
        for (const double radius : parameters.tabulated_reference_radii)
          if (radius > inner_radius && radius < outer_radius)
            full_domain_potential_radii.push_back(radius);
      std::sort(full_domain_potential_radii.begin(),
                full_domain_potential_radii.end());
      full_domain_potential_radii.erase(
        std::unique(full_domain_potential_radii.begin(),
                    full_domain_potential_radii.end()),
        full_domain_potential_radii.end());

      const unsigned int n_coefficients = sh_transform->n_coefficients();
      const unsigned int n_radii = full_domain_potential_radii.size();
      AssertDimension(surface_height_cos.size(), n_coefficients);
      AssertDimension(surface_height_sin.size(), n_coefficients);
      AssertDimension(cmb_height_cos.size(), n_coefficients);
      AssertDimension(cmb_height_sin.size(), n_coefficients);

      internal::RadialGreenMomentAccumulator internal_green_moments(
        full_domain_potential_radii,
        min_degree,
        max_degree,
        outer_radius);
      std::vector<double> source_cos_coefficients(n_coefficients, 0.0);
      std::vector<double> source_sin_coefficients(n_coefficients, 0.0);

      bool include_internal = false;
      if (include_internal_density_anomalies == "true")
        include_internal = true;
      else if (include_internal_density_anomalies == "auto")
        include_internal = true;
      if (!include_internal_sources)
        include_internal = false;

      if (include_internal)
        this->get_density_source_manager().for_each_internal_mass_source(
          reference_density_for_internal_anomalies,
          full_domain_volume_source_discretization,
          [this,
           &internal_green_moments,
           &source_cos_coefficients,
           &source_sin_coefficients]
          (const double source_mass,
           const Point<3> &point)
        {
          const double source_radius = point.norm();
          AssertThrow(source_radius > 0.0,
                      ExcMessage("Full-domain self-gravity source is undefined at radius zero."));
          const std::array<double,3> spherical_coordinates =
            aspect::Utilities::Coordinates::
            cartesian_to_spherical_coordinates(point);

          unsigned int coefficient_index = 0;
          for (unsigned int degree = min_degree;
               degree <= max_degree;
               ++degree)
            for (unsigned int order = 0;
                 order <= degree;
                 ++order, ++coefficient_index)
              {
                const std::pair<double,double> harmonics =
                  aspect::Utilities::real_spherical_harmonic(
                    degree,
                    order,
                    spherical_coordinates[2],
                    spherical_coordinates[1]);
                source_cos_coefficients[coefficient_index] =
                  source_mass * harmonics.first;
                source_sin_coefficients[coefficient_index] =
                  source_mass * harmonics.second;
              }
          internal_green_moments.add_source(source_radius,
                                            source_cos_coefficients,
                                            source_sin_coefficients);
        });

      internal_green_moments.mpi_sum(this->get_mpi_communicator());
      const std::pair<std::vector<double>, std::vector<double>>
      global_internal_coefficients = internal_green_moments.evaluate();
      const std::vector<double> &global_internal_cos =
        global_internal_coefficients.first;
      const std::vector<double> &global_internal_sin =
        global_internal_coefficients.second;

      const double surface_gravity =
        this->get_gravity_model()
        .gravity_vector(this->get_geometry_model().representative_point(0.0))
        .norm();
      AssertThrow(surface_gravity > 0.0,
                  ExcMessage("Full-domain self-gravity requires positive surface gravity."));
      full_domain_reference_gravity = surface_gravity;
      full_domain_potential_cos_coeffs.assign(
        n_radii,
        std::vector<double>(n_coefficients, 0.0));
      full_domain_potential_sin_coeffs.assign(
        n_radii,
        std::vector<double>(n_coefficients, 0.0));

      const double surface_density_contrast =
        density_below_surface - density_above_surface;
      const double cmb_density_contrast =
        density_below_cmb - density_above_cmb;

      unsigned int coefficient_index = 0;
      for (unsigned int degree = min_degree;
           degree <= max_degree;
           ++degree)
        {
          const double potential_scale =
            4.0 * numbers::PI * constants::big_g
            / (surface_gravity * (2.0 * degree + 1.0));
          for (unsigned int order = 0;
               order <= degree;
               ++order, ++coefficient_index)
            for (unsigned int radius_index = 0;
                 radius_index < n_radii;
                 ++radius_index)
              {
                const double evaluation_radius =
                  full_domain_potential_radii[radius_index];
                const double surface_kernel =
                  outer_radius
                  * std::pow(evaluation_radius / outer_radius, degree);
                const double cmb_kernel =
                  inner_radius
                  * std::pow(inner_radius / evaluation_radius,
                             degree + 1);
                const unsigned int index =
                  radius_index * n_coefficients + coefficient_index;

                full_domain_potential_cos_coeffs[radius_index]
                [coefficient_index] =
                  potential_scale
                  * (surface_density_contrast
                     * surface_height_cos[coefficient_index]
                     * surface_kernel
                     + cmb_density_contrast
                     * cmb_height_cos[coefficient_index]
                     * cmb_kernel
                     + global_internal_cos[index]);
                full_domain_potential_sin_coeffs[radius_index]
                [coefficient_index] =
                  potential_scale
                  * (surface_density_contrast
                     * surface_height_sin[coefficient_index]
                     * surface_kernel
                     + cmb_density_contrast
                     * cmb_height_sin[coefficient_index]
                     * cmb_kernel
                     + global_internal_sin[index]);
              }
        }

      // A translation of the coordinate origin by c changes the spherical
      // reference potential by -c.grad(Phi_0). Consequently its degree-one
      // amplitude is proportional to the local reference gravity, not a
      // constant boundary value. Store this correction in the same radial
      // cache used by the Stokes volume term and by every density interface.
      if (native_center_of_mass_diagnostic.valid)
        {
          const unsigned int idx10 = sh_transform->index(1, 0);
          const unsigned int idx11 = sh_transform->index(1, 1);
          const double y1_normalization =
            std::sqrt(3.0 / (4.0 * numbers::PI));
          const Tensor<1,3> &translation =
            native_center_of_mass_diagnostic.translation;

          std::vector<double> reference_cos(n_coefficients, 0.0);
          std::vector<double> reference_sin(n_coefficients, 0.0);
          reference_cos[idx10] = -translation[2] / y1_normalization;
          reference_cos[idx11] = translation[0] / y1_normalization;
          reference_sin[idx11] = translation[1] / y1_normalization;

          const GeometryModel::SphericalShell<3> &geometry =
            Plugins::get_plugin_as_type<
            const GeometryModel::SphericalShell<3>>(
              this->get_geometry_model());
          for (unsigned int radius_index = 0;
               radius_index < n_radii;
               ++radius_index)
            {
              double local_gravity = surface_gravity;
              if (radius_index + 1 != n_radii)
                {
                  const double depth =
                    std::min(geometry.maximal_depth(),
                             std::max(0.0,
                                      outer_radius
                                      - full_domain_potential_radii[
                                        radius_index]));
                  local_gravity =
                    this->get_gravity_model()
                    .gravity_vector(
                      geometry.representative_point(depth)).norm();
                }
              const double radial_scale =
                local_gravity / surface_gravity;

              for (const unsigned int index :
              {
                idx10, idx11
              })
              {
                full_domain_potential_cos_coeffs[radius_index][index] +=
                  radial_scale * reference_cos[index];
                full_domain_potential_sin_coeffs[radius_index][index] +=
                  radial_scale * reference_sin[index];
              }
            }
        }
    }


    template <>
    Tensor<1,3>
    SelfGravitation<3>::compute_internal_density_mass_dipole() const
    {
      if (!this->get_density_source_manager()
          .internal_density_anomalies_are_enabled(
            include_internal_density_anomalies))
        return Tensor<1,3>();

      return this->get_density_source_manager()
             .compute_internal_mass_moments(
               reference_density_for_internal_anomalies,
               full_domain_volume_source_discretization)
             .mass_dipole;
    }


    template <int dim>
    void
    SelfGravitation<dim>::write_native_center_of_mass_diagnostic(
      const bool /*include_current_velocity_increment*/) const
    {}


    template <int dim>
    std::pair<std::vector<double>, std::vector<double>>
    SelfGravitation<dim>::compute_internal_density_potential(const double /*outer_radius*/) const
    {
      AssertThrow(false, ExcNotImplemented());
      return std::make_pair(std::vector<double>(), std::vector<double>());
    }

    template <>
    std::pair<std::vector<double>, std::vector<double>>
    SelfGravitation<3>::compute_internal_density_potential(const double outer_radius) const
    {
      unsigned int n_coefficients = 0;
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
        n_coefficients += degree + 1;

      std::vector<double> SH_density_coecos(n_coefficients, 0.0);
      std::vector<double> SH_density_coesin(n_coefficients, 0.0);
      const auto density_source_law =
        this->get_parameters().density_source_law;

      if (density_source_law
          == Parameters<3>::Formulation::DensitySourceLaw::zero_volume_perturbation)
        return std::make_pair(SH_density_coecos, SH_density_coesin);

      // Map "auto" to either true or false depending on whether there are temperature or compositional fields
      bool actual_include_internal = false;
      if (include_internal_density_anomalies == "true")
        actual_include_internal = true;
      else if (include_internal_density_anomalies == "false")
        actual_include_internal = false;
      else if (include_internal_density_anomalies == "auto")
        {
          if (density_source_law
              == Parameters<3>::Formulation::DensitySourceLaw::legacy)
            actual_include_internal = (this->introspection().n_compositional_fields > 0 ||
                                       this->introspection().variable_exists("temperature"));
          else
            actual_include_internal = true;
        }

      if (!actual_include_internal)
        {
          return std::make_pair(SH_density_coecos, SH_density_coesin);
        }

      const unsigned int quadrature_degree = this->introspection().polynomial_degree.temperature;

      // Need to evaluate density contribution of each volume quadrature point.
      const QGauss<3> quadrature_formula(quadrature_degree);

      FEValues<3> fe_values(this->get_mapping(),
                            this->get_fe(),
                            quadrature_formula,
                            update_values |
                            update_quadrature_points |
                            update_JxW_values |
                            update_gradients);

      MaterialModel::MaterialModelInputs<3> in(fe_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<3> out(fe_values.n_quadrature_points, this->n_compositional_fields());
      this->get_density_source_manager().create_additional_material_model_outputs(out);
      in.requested_properties = MaterialModel::MaterialProperties::density;

      const double effective_tolerance =
        (internal_density_anomaly_tolerance > 0.0
         ? internal_density_anomaly_tolerance
         : 1e-12 * std::max(1.0,
                            (density_source_law
                             == Parameters<3>::Formulation::DensitySourceLaw::legacy
                             ?
                             std::abs(reference_density_for_internal_anomalies)
                             :
                             this->get_density_source_manager().get_reference_density_scale())));

      if (include_internal_density_anomalies == "auto")
        {
          double local_max_density_anomaly = 0.0;

          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                in.reinit(fe_values, cell, this->introspection(), this->get_solution());
                this->get_material_model().evaluate(in, out);

                for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                  local_max_density_anomaly =
                    std::max(local_max_density_anomaly,
                             std::abs(this->get_density_source_manager()
                                      .self_gravity_source_density(
                                        in,
                                        out,
                                        q,
                                        reference_density_for_internal_anomalies)));
              }

          const double global_max_density_anomaly =
            Utilities::MPI::max(local_max_density_anomaly,
                                this->get_mpi_communicator());

          if (global_max_density_anomaly <= effective_tolerance
              && !this->get_density_source_manager().has_internal_density_jumps())
            {
              return std::make_pair(SH_density_coecos, SH_density_coesin);
            }
        }

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            in.reinit(fe_values, cell, this->introspection(), this->get_solution());

            this->get_material_model().evaluate(in, out);

            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              {
                const double density_anomaly =
                  this->get_density_source_manager()
                  .self_gravity_source_density(
                    in,
                    out,
                    q,
                    reference_density_for_internal_anomalies);

                if (density_anomaly == 0.0)
                  continue;

                const std::array<double, 3> scoord =
                  aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(in.position[q]);
                const double r_q = in.position[q].norm();
                const double JxW = fe_values.JxW(q);

                unsigned int coefficient_index = 0;
                for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                  {
#if DEAL_II_VERSION_GTE(9,6,0)
                    const double radial_kernel =
                      (1.0 / r_q) * Utilities::pow(r_q / outer_radius, degree + 1);
#else
                    const double radial_kernel =
                      (1.0 / r_q) * std::pow(r_q / outer_radius, degree + 1);
#endif

                    for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                      {
                        const std::pair<double, double> sph_harm_vals =
                          aspect::Utilities::real_spherical_harmonic(
                            degree, order, scoord[2], scoord[1]);

                        const double weighted_density =
                          density_anomaly * radial_kernel * JxW;
                        SH_density_coecos[coefficient_index] +=
                          weighted_density * sph_harm_vals.first;
                        SH_density_coesin[coefficient_index] +=
                          weighted_density * sph_harm_vals.second;
                      }
                  }
              }
          }

      if (this->get_density_source_manager().has_internal_density_jumps())
        {
          const QGauss<2> face_quadrature_formula(quadrature_degree);
          FEFaceValues<3> face_values(this->get_mapping(),
                                      this->get_fe(),
                                      face_quadrature_formula,
                                      update_values |
                                      update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values);
          MaterialModel::MaterialModelInputs<3> face_inputs(
            face_values.n_quadrature_points,
            this->n_compositional_fields());

          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              for (const unsigned int face_no : cell->face_indices())
                {
                  if (cell->at_boundary(face_no)
                      || cell->has_periodic_neighbor(face_no))
                    continue;

                  const auto neighbor = cell->neighbor(face_no);
                  const Point<3> inner_cell_center =
                    this->get_density_source_manager()
                    .radial_cell_representative_point(cell);
                  const Point<3> outer_cell_center =
                    this->get_density_source_manager()
                    .radial_cell_representative_point(neighbor);
                  if (inner_cell_center.norm() >= outer_cell_center.norm())
                    continue;

                  const auto face = cell->face(face_no);
                  const double density_contrast =
                    this->get_density_source_manager()
                    .internal_density_jump_across_face(
                      inner_cell_center,
                      outer_cell_center,
                      face->vertex(0).norm());
                  if (density_contrast == 0.0)
                    continue;

                  bool all_vertices_match = true;
                  for (unsigned int vertex = 1;
                       vertex < face->n_vertices();
                       ++vertex)
                    all_vertices_match =
                      all_vertices_match
                      && (this->get_density_source_manager()
                          .internal_density_jump_across_face(
                            inner_cell_center,
                            outer_cell_center,
                            face->vertex(vertex).norm())
                          == density_contrast);
                  if (!all_vertices_match)
                    continue;

                  face_values.reinit(cell, face_no);

                  face_inputs.reinit(face_values,
                                     cell,
                                     this->introspection(),
                                     this->get_solution());

                  for (unsigned int q = 0;
                       q < face_values.n_quadrature_points;
                       ++q)
                    {
                      const Point<3> point = face_inputs.position[q];
                      const double radius = point.norm();
                      AssertThrow(radius > 0.0,
                                  ExcMessage("Internal density jump potential is undefined at radius zero."));
                      const double surface_density =
                        density_contrast
                        * this->get_density_source_manager()
                        .mechanical_radial_displacement(face_inputs, q);
                      const std::array<double,3> spherical_coordinates =
                        aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(point);
                      const double JxW = face_values.JxW(q);

                      unsigned int coefficient_index = 0;
                      for (unsigned int degree = min_degree;
                           degree <= max_degree;
                           ++degree)
                        {
#if DEAL_II_VERSION_GTE(9,6,0)
                          const double radial_kernel =
                            (1.0 / radius)
                            * Utilities::pow(radius / outer_radius, degree + 1);
#else
                          const double radial_kernel =
                            (1.0 / radius)
                            * std::pow(radius / outer_radius, degree + 1);
#endif
                          for (unsigned int order = 0;
                               order <= degree;
                               ++order, ++coefficient_index)
                            {
                              const std::pair<double,double> spherical_harmonic =
                                aspect::Utilities::real_spherical_harmonic(
                                  degree,
                                  order,
                                  spherical_coordinates[2],
                                  spherical_coordinates[1]);
                              const double weighted_surface_density =
                                surface_density * radial_kernel * JxW;
                              SH_density_coecos[coefficient_index] +=
                                weighted_surface_density * spherical_harmonic.first;
                              SH_density_coesin[coefficient_index] +=
                                weighted_surface_density * spherical_harmonic.second;
                            }
                        }
                    }
                }
        }

      dealii::Utilities::MPI::sum(SH_density_coecos, this->get_mpi_communicator(), SH_density_coecos);
      dealii::Utilities::MPI::sum(SH_density_coesin, this->get_mpi_communicator(), SH_density_coesin);

      return std::make_pair(SH_density_coecos, SH_density_coesin);
    }

    template <int dim>
    std::pair<std::pair<double, std::pair<std::vector<double>, std::vector<double>>>, std::pair<double, std::pair<std::vector<double>, std::vector<double>>>>
    SelfGravitation<dim>::compute_topography_potential(const double /*outer_radius*/, const double /*inner_radius*/) const
    {
      AssertThrow(false, ExcNotImplemented());
      std::pair<double, std::pair<std::vector<double>, std::vector<double>>> temp;
      return std::make_pair(temp, temp);
    }

    template <>
    std::pair<std::pair<double, std::pair<std::vector<double>, std::vector<double>>>, std::pair<double, std::pair<std::vector<double>, std::vector<double>>>>
    SelfGravitation<3>::compute_topography_potential(const double outer_radius, const double inner_radius) const
    {
      const Postprocess::BoundaryDensities<3> &boundary_densities =
        this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::BoundaryDensities<3>>();

      const double top_layer_average_density = boundary_densities.density_at_top();
      const double bottom_layer_average_density = boundary_densities.density_at_bottom();

      const types::boundary_id top_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      const unsigned int quadrature_degree = this->introspection().polynomial_degree.temperature;
      const QGauss<2> quadrature_formula_face(quadrature_degree);

      FEFaceValues<3> fe_face_values(this->get_mapping(),
                                     this->get_fe(),
                                     quadrature_formula_face,
                                     update_values |
                                     update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

      std::vector<std::pair<Point<3>, std::pair<double, double>>> surface_stored_values;
      std::vector<std::pair<Point<3>, std::pair<double, double>>> CMB_stored_values;

      const bool use_free_surface_topography = true;
      const bool use_free_CMB_topography = include_cmb_contribution;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            unsigned int face_idx = numbers::invalid_unsigned_int;
            bool at_upper_surface = false;
            {
              for (const unsigned int f : cell->face_indices())
                {
                  if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
                    {
                      face_idx = f;
                      at_upper_surface = true;
                      break;
                    }
                  else if (cell->at_boundary(f) && cell->face(f)->boundary_id() == bottom_boundary_id)
                    {
                      face_idx = f;
                      at_upper_surface = false;
                      break;
                    }
                }
              if (face_idx == numbers::invalid_unsigned_int)
                continue;
            }

            fe_face_values.reinit(cell, face_idx);

            if (at_upper_surface)
              {
                if (use_free_surface_topography)
                  {
                    const auto &boundary_traction_manager = this->get_boundary_traction_manager();
                    const std::set<types::boundary_id> &prescribed_traction_boundary_indicators =
                      boundary_traction_manager.get_prescribed_boundary_traction_indicators();
                    const bool has_active_boundary_traction = (boundary_traction_manager.get_active_plugins().empty() == false);

                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        const Point<3> current_position = fe_face_values.quadrature_point(q);
                        double topography = this->get_geometry_model().height_above_reference_surface(current_position);

                        if (has_active_boundary_traction &&
                            prescribed_traction_boundary_indicators.find(cell->face(face_idx)->boundary_id()) !=
                            prescribed_traction_boundary_indicators.end())
                          {
                            const Tensor<1, 3> traction = boundary_traction_manager.boundary_traction(
                                                            cell->face(face_idx)->boundary_id(), fe_face_values.quadrature_point(q), fe_face_values.normal_vector(q));
                            const double normal_traction = traction * fe_face_values.normal_vector(q);

                            if (std::abs(normal_traction) > 1e-10)
                              {
                                const double gravity = this->get_gravity_model().gravity_vector(current_position).norm();
                                const double delta_rho = top_layer_average_density;
                                if (std::abs(delta_rho) > 0.0)
                                  topography -= normal_traction / (gravity * delta_rho);
                              }
                          }

                        surface_stored_values.emplace_back(current_position, std::make_pair(fe_face_values.JxW(q), topography));
                      }
                  }
                else
                  {
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        surface_stored_values.emplace_back(fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), 0.0));
                      }
                  }
              }

            if (at_upper_surface == false)
              {
                if (use_free_CMB_topography)
                  {
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        const Point<3> current_position = fe_face_values.quadrature_point(q);
                        const double topography = this->get_geometry_model().height_above_reference_surface(current_position) + (outer_radius - inner_radius);
                        CMB_stored_values.emplace_back(current_position, std::make_pair(fe_face_values.JxW(q), topography));
                      }
                  }
                else
                  {
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        CMB_stored_values.emplace_back(fe_face_values.quadrature_point(q), std::make_pair(fe_face_values.JxW(q), 0.0));
                      }
                  }
              }
          }

      std::vector<std::vector<double>> surface_topo_spherical_function;
      std::vector<std::vector<double>> CMB_topo_spherical_function;

      for (const auto &surface_stored_value : surface_stored_values)
        {
          const std::array<double, 3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(surface_stored_value.first);
          const double infinitesimal = surface_stored_value.second.first / (outer_radius * outer_radius);
          surface_topo_spherical_function.emplace_back(std::vector<double> {scoord[2],
                                                                            scoord[1],
                                                                            infinitesimal,
                                                                            surface_stored_value.second.second
                                                                           });
        }

      for (const auto &CMB_stored_value : CMB_stored_values)
        {
          const std::array<double, 3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(CMB_stored_value.first);
          const double infinitesimal = CMB_stored_value.second.first / (inner_radius * inner_radius);
          CMB_topo_spherical_function.emplace_back(std::vector<double> {scoord[2],
                                                                        scoord[1],
                                                                        infinitesimal,
                                                                        CMB_stored_value.second.second
                                                                       });
        }

      std::pair<double, std::pair<std::vector<double>, std::vector<double>>> SH_surface_topo_coes
        = std::make_pair(top_layer_average_density, to_spherical_harmonic_coefficients(surface_topo_spherical_function));
      std::pair<double, std::pair<std::vector<double>, std::vector<double>>> SH_CMB_topo_coes
        = std::make_pair(bottom_layer_average_density, to_spherical_harmonic_coefficients(CMB_topo_spherical_function));

      return std::make_pair(SH_surface_topo_coes, SH_CMB_topo_coes);
    }
  }
}


// Explicit instantiations
namespace aspect
{
  namespace PotentialFeedback
  {
    template class SelfGravitation<2>;
    template class SelfGravitation<3>;
  }
}
