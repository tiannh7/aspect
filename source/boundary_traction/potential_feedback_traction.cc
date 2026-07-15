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

#include <algorithm>
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
    PotentialFeedbackTraction<dim>::initialize()
    {
      if (&primary_provider() != this)
        return;

      const double mm_initial_elastic_dt = this->get_material_model().initial_elastic_time_step();
      settings.initial_displacement_timestep = (mm_initial_elastic_dt > 0.0) ? mm_initial_elastic_dt : 0.0;

      if (self_gravity_active)
        self_gravity.configure_from_potential_feedback_settings(settings);
      if (rotational_feedback_active)
        rotational_feedback.configure_from_potential_feedback_settings(settings);
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
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::update()
    {
      if (&primary_provider() != this)
        return;

      if (glacial_isostatic_adjustment_active)
        glacial_isostatic_adjustment.update();

      if (self_gravity_active)
        self_gravity.update();

      if (rotational_feedback_active)
        rotational_feedback.update();

      if (glacial_isostatic_adjustment_active)
        glacial_isostatic_adjustment.update_load_from_current_potential();
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

      return (!self_gravity_active || self_gravity.potential_is_converged())
             &&
             (!rotational_feedback_active
              || rotational_feedback.potential_is_converged())
             &&
             (!glacial_isostatic_adjustment_active
              || glacial_isostatic_adjustment.potential_is_converged());
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
      "feedback'' parameter hierarchy and dispatches the active self-gravity "
      "and rotational-feedback mechanisms without requiring legacy per-plugin "
      "parameter blocks.")
  }
}
