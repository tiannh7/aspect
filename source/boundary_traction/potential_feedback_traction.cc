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

#include <aspect/boundary_traction/potential_feedback_traction.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/simulator_signals.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace BoundaryTraction
  {
    template <int dim>
    bool
    PotentialFeedbackTraction<dim>::mechanism_is_active(
      const std::string &name) const
    {
      return std::find(settings.feedback_mechanisms.begin(),
                       settings.feedback_mechanisms.end(),
                       name) != settings.feedback_mechanisms.end();
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::
    set_active_feedback_boundaries_from_traction_model()
    {
      settings.active_feedback_boundary_indicators.clear();
      settings.include_surface_feedback = false;
      settings.include_cmb_feedback = false;

      const auto &traction_manager = this->get_boundary_traction_manager();
      const auto &plugin_names = traction_manager.get_active_plugin_names();
      const auto &plugin_boundaries =
        traction_manager.get_active_plugin_boundary_indicators();

      Assert(plugin_names.size() == plugin_boundaries.size(),
             ExcInternalError());

      for (unsigned int plugin_index = 0;
           plugin_index < plugin_names.size();
           ++plugin_index)
        if (plugin_names[plugin_index] == "potential feedback")
          settings.active_feedback_boundary_indicators.insert(
            plugin_boundaries[plugin_index]);

      const types::boundary_id top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      settings.include_surface_feedback =
        settings.active_feedback_boundary_indicators.count(top_boundary_id) > 0;
      settings.include_cmb_feedback =
        settings.active_feedback_boundary_indicators.count(bottom_boundary_id) > 0;

      AssertThrow(settings.include_surface_feedback
                  || settings.include_cmb_feedback,
                  ExcMessage("The `potential feedback' boundary traction model "
                             "must be prescribed on at least the top/surface or "
                             "bottom/CMB boundary."));
    }



    template <int dim>
    PotentialFeedbackTraction<dim> &
    PotentialFeedbackTraction<dim>::primary_provider()
    {
      for (const auto &plugin :
           this->get_boundary_traction_manager().get_active_plugins())
        if (auto *provider =
              dynamic_cast<PotentialFeedbackTraction<dim> *>(plugin.get()))
          return *provider;

      AssertThrow(false,
                  ExcMessage("Could not find a primary `potential feedback' "
                             "boundary traction provider."));
      return *this;
    }



    template <int dim>
    const PotentialFeedbackTraction<dim> &
    PotentialFeedbackTraction<dim>::primary_provider() const
    {
      for (const auto &plugin :
           this->get_boundary_traction_manager().get_active_plugins())
        if (const auto *provider =
              dynamic_cast<const PotentialFeedbackTraction<dim> *>(
                plugin.get()))
          return *provider;

      AssertThrow(false,
                  ExcMessage("Could not find a primary `potential feedback' "
                             "boundary traction provider."));
      return *this;
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::reset_velocity_update_reference()
    {
      const unsigned int velocity_block =
        this->introspection().block_indices.velocities;
      const LinearAlgebra::Vector &current_velocity =
        this->get_solution().block(velocity_block);

      previous_stokes_velocity.reinit(
        this->introspection().index_sets.system_partitioning[velocity_block],
        this->get_mpi_communicator());
      previous_stokes_velocity = current_velocity;
      velocity_update_relative_change =
        std::numeric_limits<double>::infinity();
      velocity_update_reference_is_initialized = true;
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::update_velocity_update_diagnostic()
    {
      if (&primary_provider() != this)
        return;

      if (!velocity_update_reference_is_initialized)
        reset_velocity_update_reference();

      const unsigned int velocity_block =
        this->introspection().block_indices.velocities;
      LinearAlgebra::Vector current_velocity(
        this->introspection().index_sets.system_partitioning[velocity_block],
        this->get_mpi_communicator());
      current_velocity = this->get_solution().block(velocity_block);
      LinearAlgebra::Vector velocity_difference(current_velocity);
      velocity_difference.add(-1.0, previous_stokes_velocity);

      const double velocity_norm = current_velocity.l2_norm();
      const double difference_norm = velocity_difference.l2_norm();
      velocity_update_relative_change =
        (velocity_norm == 0.0 && difference_norm == 0.0)
        ? 0.0
        : difference_norm
        / std::max(velocity_norm, std::numeric_limits<double>::min());

      previous_stokes_velocity = current_velocity;

      if (settings.convergence_criterion_is_active("velocity update"))
        this->get_pcout()
            << "      Potential-feedback velocity update: relative L2 change="
            << std::scientific << std::setprecision(6)
            << velocity_update_relative_change
            << std::defaultfloat << std::endl;
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::write_polar_wander_timing_diagnostic(
      const std::string &stage) const
    {
      const char *enabled = std::getenv("ASPECT_PW_TIMING_DEBUG");
      if (enabled == nullptr || std::atof(enabled) == 0.0)
        return;

      if (&primary_provider() != this)
        return;

      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) != 0)
        return;

      const auto coefficient_or_zero =
        [](const bool active,
           const std::function<std::pair<double,double>()> &coefficient)
      {
        return active ? coefficient() : std::make_pair(0.0, 0.0);
      };

      const std::pair<double,double> self_surface =
        coefficient_or_zero(self_gravity_active,
                            [this]()
      {
        return self_gravity.total_surface_potential_coefficient(2, 1);
      });
      const std::pair<double,double> self_cmb =
        coefficient_or_zero(self_gravity_active,
                            [this]()
      {
        return self_gravity.cmb_mass_potential_coefficient(2, 1);
      });
      const std::pair<double,double> rotational_surface =
        coefficient_or_zero(rotational_feedback_active,
                            [this]()
      {
        return rotational_feedback.surface_potential_coefficient(2, 1);
      });
      const std::pair<double,double> rotational_cmb =
        coefficient_or_zero(rotational_feedback_active,
                            [this]()
      {
        return rotational_feedback.cmb_potential_coefficient(2, 1);
      });

      double full_surface_phi = 0.0;
      double full_cmb_phi = 0.0;
      double full_surface_height = 0.0;
      double full_cmb_height = 0.0;
      double full_surface_l21_cosine = 0.0;
      double full_surface_l21_sine = 0.0;
      double full_cmb_l21_cosine = 0.0;
      double full_cmb_l21_sine = 0.0;
      double boundary_surface_radial_traction = 0.0;
      double boundary_cmb_radial_traction = 0.0;
      double gia_surface_mass_density_value = 0.0;

      if constexpr (dim == 3)
        {
          const GeometryModel::SphericalShell<dim> &geometry =
            Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
              this->get_geometry_model());
          const double outer_radius = geometry.outer_radius();
          const double inner_radius = geometry.inner_radius();
          const Point<dim> unit_point(1.0/std::sqrt(2.0),
                                      0.0,
                                      1.0/std::sqrt(2.0));
          const Point<dim> surface_point = outer_radius * unit_point;
          const Point<dim> cmb_point = inner_radius * unit_point;
          const Tensor<1,dim> surface_normal = unit_point;
          const Tensor<1,dim> cmb_normal = -unit_point;
          const types::boundary_id top_boundary_id =
            this->get_geometry_model()
            .translate_symbolic_boundary_name_to_id("top");
          const types::boundary_id bottom_boundary_id =
            this->get_geometry_model()
            .translate_symbolic_boundary_name_to_id("bottom");

          full_surface_phi = full_domain_potential(surface_point);
          full_cmb_phi = full_domain_potential(cmb_point);
          const double surface_g =
            this->get_gravity_model().gravity_vector(surface_point).norm();
          const double cmb_g =
            this->get_gravity_model().gravity_vector(cmb_point).norm();
          full_surface_height =
            surface_g > 0.0 ? full_surface_phi / surface_g : 0.0;
          full_cmb_height = cmb_g > 0.0 ? full_cmb_phi / cmb_g : 0.0;

          constexpr unsigned int n_theta = 64;
          constexpr unsigned int n_phi = 128;
          const double dtheta = numbers::PI / n_theta;
          const double dphi = 2.0 * numbers::PI / n_phi;
          for (unsigned int theta_index = 0;
               theta_index < n_theta;
               ++theta_index)
            {
              const double theta = (theta_index + 0.5) * dtheta;
              const double sin_theta = std::sin(theta);
              const double cos_theta = std::cos(theta);
              for (unsigned int phi_index = 0;
                   phi_index < n_phi;
                   ++phi_index)
                {
                  const double phi = (phi_index + 0.5) * dphi;
                  const Point<dim> direction(sin_theta * std::cos(phi),
                                             sin_theta * std::sin(phi),
                                             cos_theta);
                  const std::pair<double,double> y21 =
                    Utilities::real_spherical_harmonic(2, 1, theta, phi);
                  const double weight = sin_theta * dtheta * dphi;
                  const Point<dim> surface_projection_point =
                    outer_radius * direction;
                  const Point<dim> cmb_projection_point =
                    inner_radius * direction;
                  const double surface_projection_g =
                    this->get_gravity_model()
                    .gravity_vector(surface_projection_point).norm();
                  const double cmb_projection_g =
                    this->get_gravity_model()
                    .gravity_vector(cmb_projection_point).norm();
                  const double surface_height =
                    surface_projection_g > 0.0
                    ? full_domain_potential(surface_projection_point)
                    / surface_projection_g
                    : 0.0;
                  const double cmb_height =
                    cmb_projection_g > 0.0
                    ? full_domain_potential(cmb_projection_point)
                    / cmb_projection_g
                    : 0.0;

                  full_surface_l21_cosine +=
                    surface_height * y21.first * weight;
                  full_surface_l21_sine +=
                    surface_height * y21.second * weight;
                  full_cmb_l21_cosine += cmb_height * y21.first * weight;
                  full_cmb_l21_sine += cmb_height * y21.second * weight;
                }
            }

          boundary_surface_radial_traction =
            boundary_traction(top_boundary_id,
                              surface_point,
                              surface_normal) * surface_normal;
          boundary_cmb_radial_traction =
            boundary_traction(bottom_boundary_id,
                              cmb_point,
                              cmb_normal) * cmb_normal;
          if (glacial_isostatic_adjustment_active)
            gia_surface_mass_density_value =
              glacial_isostatic_adjustment.surface_mass_density(surface_point);
        }

      static unsigned int diagnostic_call = 0;
      const std::string filename =
        this->get_parameters().output_directory
        + "/aspect_polar_wander_timing_diagnostic";
      std::ofstream output(filename,
                           diagnostic_call == 0 ? std::ios::out
                           : std::ios::app);
      if (!output)
        return;

      if (diagnostic_call == 0)
        output
            << "# ASPECT l2m1 polar-wander timing diagnostic\n"
            << "# Coefficients are Phi/g spherical-harmonic coefficients.\n"
            << "# columns: call timestep time stage "
            << "self_total_surface_cos self_total_surface_sin self_cmb_cos self_cmb_sin "
            << "rot_surface_cos rot_surface_sin rot_cmb_cos rot_cmb_sin "
            << "total_surface_cos total_surface_sin total_cmb_cos total_cmb_sin "
            << "full_surface_phi full_cmb_phi full_surface_height full_cmb_height "
            << "full_surface_l21_cos full_surface_l21_sin "
            << "full_cmb_l21_cos full_cmb_l21_sin "
            << "boundary_surface_radial_traction boundary_cmb_radial_traction "
            << "gia_surface_mass_density self_gravity_rel_change "
            << "rotational_rel_change potential_converged\n";

      output << std::setprecision(16) << std::scientific
             << diagnostic_call << ' '
             << this->get_timestep_number() << ' '
             << this->get_time() << ' '
             << stage << ' '
             << self_surface.first << ' ' << self_surface.second << ' '
             << self_cmb.first << ' ' << self_cmb.second << ' '
             << rotational_surface.first << ' '
             << rotational_surface.second << ' '
             << rotational_cmb.first << ' ' << rotational_cmb.second << ' '
             << self_surface.first + rotational_surface.first << ' '
             << self_surface.second + rotational_surface.second << ' '
             << self_cmb.first + rotational_cmb.first << ' '
             << self_cmb.second + rotational_cmb.second << ' '
             << full_surface_phi << ' ' << full_cmb_phi << ' '
             << full_surface_height << ' ' << full_cmb_height << ' '
             << full_surface_l21_cosine << ' '
             << full_surface_l21_sine << ' '
             << full_cmb_l21_cosine << ' '
             << full_cmb_l21_sine << ' '
             << boundary_surface_radial_traction << ' '
             << boundary_cmb_radial_traction << ' '
             << gia_surface_mass_density_value << ' '
             << (self_gravity_active
                 ? self_gravity.potential_relative_change_value()
                 : 0.0) << ' '
             << (rotational_feedback_active
                 ? rotational_feedback.potential_relative_change_value()
                 : 0.0) << ' '
             << (potential_is_converged() ? 1 : 0)
             << '\n';
      ++diagnostic_call;
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::initialize()
    {
      if (&primary_provider() != this)
        return;

      const double mm_initial_elastic_dt = this->get_material_model().initial_elastic_time_step();
      settings.initial_displacement_timestep = (mm_initial_elastic_dt > 0.0) ? mm_initial_elastic_dt : 0.0;

      if (self_gravity_active)
        self_gravity.configure_from_potential_feedback_settings(settings);
      if (rotational_feedback_active)
        {
          rotational_feedback.configure_from_potential_feedback_settings(settings);
          if (self_gravity_active)
            rotational_feedback
            .set_self_gravity_surface_potential_coefficient_function(
              [this](const unsigned int degree, const unsigned int order)
            {
              return self_gravity.total_surface_potential_coefficient(degree,
                                                                      order);
            });
        }
      if (glacial_isostatic_adjustment_active)
        glacial_isostatic_adjustment
        .configure_from_potential_feedback_settings(settings);

      if (glacial_isostatic_adjustment_active)
        {
          const auto gia_traction =
            [this](const types::boundary_id boundary_indicator,
                   const Point<dim> &position,
                   const Tensor<1,dim> &normal_vector)
          {
            return glacial_isostatic_adjustment.boundary_traction(
                     boundary_indicator, position, normal_vector);
          };

          self_gravity.set_additional_load_traction_function(gia_traction);
          if (rotational_feedback_active)
            rotational_feedback.set_additional_load_traction_function(
              gia_traction);

          glacial_isostatic_adjustment
          .set_surface_potential_height_function(
            [this](const Point<dim> &position)
          {
            return this->surface_potential_height(position);
          });
        }

      if (self_gravity_active)
        self_gravity.initialize();

      if (rotational_feedback_active)
        rotational_feedback.initialize();

      if (glacial_isostatic_adjustment_active)
        glacial_isostatic_adjustment.initialize();

      this->get_signals().post_stokes_solver.connect(
        [this](const SimulatorAccess<dim> &,
               const unsigned int,
               const unsigned int,
               const SolverControl &,
               const SolverControl &)
      {
        this->update_velocity_update_diagnostic();
        this->write_polar_wander_timing_diagnostic(
          "after_post_stokes_feedback_signals");
      });
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::update()
    {
      if (&primary_provider() != this)
        return;

      reset_velocity_update_reference();

      write_polar_wander_timing_diagnostic("before_update");

      if (glacial_isostatic_adjustment_active)
        glacial_isostatic_adjustment.update();

      write_polar_wander_timing_diagnostic("after_gia_update");

      if (self_gravity_active)
        self_gravity.update();

      write_polar_wander_timing_diagnostic("after_self_gravity_update");

      if (rotational_feedback_active)
        rotational_feedback.update();

      write_polar_wander_timing_diagnostic("after_rotational_update");

      if (glacial_isostatic_adjustment_active)
        glacial_isostatic_adjustment.update_load_from_current_potential();

      write_polar_wander_timing_diagnostic("after_gia_load_from_current_potential");
    }



    template <int dim>
    Tensor<1,dim>
    PotentialFeedbackTraction<dim>::
    boundary_traction(const types::boundary_id boundary_indicator,
                      const Point<dim> &position,
                      const Tensor<1,dim> &normal_vector) const
    {
      if (&primary_provider() != this)
        return primary_provider().boundary_traction(boundary_indicator,
                                                    position,
                                                    normal_vector);

      Tensor<1,dim> traction;

      if (glacial_isostatic_adjustment_active)
        traction += glacial_isostatic_adjustment.boundary_traction(
                      boundary_indicator, position, normal_vector);

      if (self_gravity_active)
        traction += self_gravity.boundary_traction(boundary_indicator,
                                                   position,
                                                   normal_vector);

      if (rotational_feedback_active)
        traction += rotational_feedback.boundary_traction(boundary_indicator,
                                                          position,
                                                          normal_vector);

      return traction;
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::surface_potential_height(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().surface_potential_height(position);

      const types::boundary_id top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      double potential_height = 0.0;
      if (self_gravity_active)
        potential_height += self_gravity.potential_height(top_boundary_id,
                                                          position);
      if (rotational_feedback_active)
        potential_height += rotational_feedback.potential_height(
                              top_boundary_id, position);
      return potential_height;
    }



    template <int dim>
    bool
    PotentialFeedbackTraction<dim>::has_glacial_isostatic_adjustment() const
    {
      if (&primary_provider() != this)
        return primary_provider().has_glacial_isostatic_adjustment();
      return glacial_isostatic_adjustment_active;
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_surface_mass_density(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_surface_mass_density(position);
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.surface_mass_density(position);
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_ice_load_mass_density(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_ice_load_mass_density(position);
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.ice_load_mass_density(position);
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_ocean_load_mass_density(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_ocean_load_mass_density(position);
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.ocean_load_mass_density(position);
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_sea_level_change(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_sea_level_change(position);
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.sea_level_change(position);
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_ocean_function(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_ocean_function(position);
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.ocean_function(position);
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_barystatic_sea_level() const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_barystatic_sea_level();
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.barystatic_sea_level();
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::gia_eustatic_sea_level() const
    {
      if (&primary_provider() != this)
        return primary_provider().gia_eustatic_sea_level();
      AssertThrow(glacial_isostatic_adjustment_active,
                  ExcMessage("GIA is not active."));
      return glacial_isostatic_adjustment.eustatic_sea_level();
    }



    template <int dim>
    bool
    PotentialFeedbackTraction<dim>::has_self_gravity_feedback() const
    {
      if (&primary_provider() != this)
        return primary_provider().has_self_gravity_feedback();

      return self_gravity_active;
    }



    template <int dim>
    std::pair<double,double>
    PotentialFeedbackTraction<dim>::surface_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      if (&primary_provider() != this)
        return primary_provider().surface_mass_potential_coefficient(degree,
                                                                     order);

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.surface_mass_potential_coefficient(degree, order);
    }


    template <int dim>
    std::pair<double,double>
    PotentialFeedbackTraction<dim>::external_load_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      if (&primary_provider() != this)
        return primary_provider().external_load_surface_potential_coefficient(
                 degree, order);

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.external_load_surface_potential_coefficient(degree,
                                                                      order);
    }



    template <int dim>
    std::pair<double,double>
    PotentialFeedbackTraction<dim>::surface_deformation_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      if (&primary_provider() != this)
        return primary_provider().surface_deformation_mass_potential_coefficient(
                 degree, order);

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.surface_deformation_mass_potential_coefficient(
               degree, order);
    }



    template <int dim>
    std::pair<double,double>
    PotentialFeedbackTraction<dim>::cmb_mass_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      if (&primary_provider() != this)
        return primary_provider().cmb_mass_potential_coefficient(degree,
                                                                 order);

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.cmb_mass_potential_coefficient(degree, order);
    }



    template <int dim>
    std::pair<double,double>
    PotentialFeedbackTraction<dim>::tidal_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      if (&primary_provider() != this)
        return primary_provider().tidal_surface_potential_coefficient(degree,
                                                                      order);

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.tidal_surface_potential_coefficient(degree, order);
    }


    template <int dim>
    std::pair<double,double>
    PotentialFeedbackTraction<dim>::rotational_surface_potential_coefficient(
      const unsigned int degree,
      const unsigned int order) const
    {
      if (&primary_provider() != this)
        return primary_provider().rotational_surface_potential_coefficient(
                 degree, order);

      if (!rotational_feedback_active)
        return {0.0, 0.0};

      return rotational_feedback.surface_potential_coefficient(degree, order);
    }



    template <int dim>
    Tensor<1,dim>
    PotentialFeedbackTraction<dim>::reference_frame_body_force(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().reference_frame_body_force(position);

      Tensor<1,dim> force;
      if (self_gravity_active)
        force += self_gravity.reference_frame_body_force(position);

      return force;
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::full_domain_potential(
      const Point<dim> &position) const
    {
      if (&primary_provider() != this)
        return primary_provider().full_domain_potential(position);

      double potential = 0.0;
      if (self_gravity_active)
        {
          if (self_gravity.has_full_domain_potential())
            potential += self_gravity.full_domain_potential(position);
        }
      if (rotational_feedback_active)
        potential += rotational_feedback.full_domain_potential(position);

      return potential;
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::surface_density_jump() const
    {
      if (&primary_provider() != this)
        return primary_provider().surface_density_jump();

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.surface_density_jump();
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::cmb_density_jump() const
    {
      if (&primary_provider() != this)
        return primary_provider().cmb_density_jump();

      AssertThrow(self_gravity_active,
                  ExcMessage("The `potential feedback' boundary traction "
                             "does not have active self-gravity feedback."));
      return self_gravity.cmb_density_jump();
    }



    template <int dim>
    bool
    PotentialFeedbackTraction<dim>::potential_is_converged() const
    {
      if (&primary_provider() != this)
        return primary_provider().potential_is_converged();

      const bool velocity_update_is_converged =
        !settings.convergence_criterion_is_active("velocity update")
        || !settings.iterate_with_stokes
        || (settings.freeze_feedback_after_timestep_zero
            && this->get_timestep_number() > 0)
        || velocity_update_relative_change
        <= settings.velocity_update_relative_tolerance;

      return velocity_update_is_converged
             &&
             (!self_gravity_active || self_gravity.potential_is_converged())
             &&
             (!rotational_feedback_active
              || rotational_feedback.potential_is_converged())
             &&
             (!glacial_isostatic_adjustment_active
              || glacial_isostatic_adjustment.potential_is_converged());
    }



    template <int dim>
    double
    PotentialFeedbackTraction<dim>::
    velocity_update_relative_change_value() const
    {
      if (&primary_provider() != this)
        return primary_provider().velocity_update_relative_change_value();
      return velocity_update_relative_change;
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::declare_parameters(ParameterHandler &prm)
    {
      PotentialFeedback::Settings::declare_parameters(prm);
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::parse_parameters(ParameterHandler &prm)
    {
      settings.parse_parameters(prm);
      set_active_feedback_boundaries_from_traction_model();

      self_gravity_active = mechanism_is_active("self gravity")
                            || mechanism_is_active("tidal potential");
      rotational_feedback_active =
        mechanism_is_active("rotational feedback");
      glacial_isostatic_adjustment_active =
        mechanism_is_active("glacial isostatic adjustment");

      AssertThrow(settings.has_active_mechanisms(),
                  ExcMessage("The `potential feedback' boundary traction "
                             "plugin requires at least one active mechanism "
                             "in `Potential feedback/List of feedback "
                             "mechanisms'."));

      if (self_gravity_active)
        {
          self_gravity.initialize_simulator(this->get_simulator());
          self_gravity.configure_from_potential_feedback_settings(settings);
        }

      if (rotational_feedback_active)
        {
          rotational_feedback.initialize_simulator(this->get_simulator());
          rotational_feedback.configure_from_potential_feedback_settings(
            settings);
        }

      if (glacial_isostatic_adjustment_active)
        {
          AssertThrow(settings.include_surface_feedback,
                      ExcMessage("Glacial isostatic adjustment requires the "
                                 "`potential feedback' traction model on the "
                                 "top boundary."));
          glacial_isostatic_adjustment.initialize_simulator(
            this->get_simulator());
          glacial_isostatic_adjustment
          .configure_from_potential_feedback_settings(settings);
        }
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::save(
      std::map<std::string, std::string> &status_strings) const
    {
      if (&primary_provider() != this
          || !glacial_isostatic_adjustment_active)
        return;

      std::ostringstream output;
      {
        aspect::oarchive archive(output);
        archive << glacial_isostatic_adjustment;
      }
      status_strings["PotentialFeedbackTractionGIA"] = output.str();
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::load(
      const std::map<std::string, std::string> &status_strings)
    {
      if (&primary_provider() != this
          || !glacial_isostatic_adjustment_active)
        return;

      const auto state = status_strings.find("PotentialFeedbackTractionGIA");
      if (state != status_strings.end())
        {
          std::istringstream input(state->second);
          aspect::iarchive archive(input);
          archive >> glacial_isostatic_adjustment;
        }
    }
  }
}

namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(
      PotentialFeedbackTraction,
      "potential feedback",
      "Unified boundary traction model for potential-feedback-derived normal "
      "traction. The model is configured through the shared ``Potential "
      "feedback'' parameter hierarchy and dispatches the active self-gravity, "
      "rotational-feedback, and glacial-isostatic-adjustment mechanisms "
      "without requiring legacy per-plugin parameter blocks.")
  }
}
