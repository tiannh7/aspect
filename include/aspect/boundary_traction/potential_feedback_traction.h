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

#ifndef _aspect_boundary_traction_potential_feedback_traction_h
#define _aspect_boundary_traction_potential_feedback_traction_h

#include <aspect/boundary_traction/interface.h>
#include <aspect/potential_feedback/glacial_isostatic_adjustment.h>
#include <aspect/potential_feedback/rotational_feedback.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/potential_feedback/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace BoundaryTraction
  {
    /**
     * Thin boundary-traction adapter for potential-feedback-derived normal
     * traction.
     *
     * This class implements the user-facing `potential feedback' boundary
     * traction model and dispatches self-gravity, tidal, rotational, and GIA
     * mechanisms configured through the shared `Potential feedback'
     * hierarchy.
     *
     * @ingroup BoundaryTractions
     */
    template <int dim>
    class PotentialFeedbackTraction : public Interface<dim>,
      public ::aspect::SimulatorAccess<dim>,
      public PotentialFeedback::BoundaryTractionMarker
    {
      public:
        void initialize() override;

        void update() override;

        Tensor<1,dim>
        boundary_traction(const types::boundary_id boundary_indicator,
                          const Point<dim> &position,
                          const Tensor<1,dim> &normal_vector) const override;

        double surface_potential_height(const Point<dim> &position) const;

        bool has_glacial_isostatic_adjustment() const;

        double gia_surface_mass_density(const Point<dim> &position) const;

        double gia_ice_load_mass_density(const Point<dim> &position) const;

        double gia_ocean_load_mass_density(const Point<dim> &position) const;

        double gia_sea_level_change(const Point<dim> &position) const;

        double gia_ocean_function(const Point<dim> &position) const;

        double gia_barystatic_sea_level() const;

        double gia_eustatic_sea_level() const;

        bool has_self_gravity_feedback() const;

        std::pair<double,double>
        surface_mass_potential_coefficient(const unsigned int degree,
                                           const unsigned int order) const;

        std::pair<double,double>
        external_load_surface_potential_coefficient(const unsigned int degree,
                                                    const unsigned int order) const;

        std::pair<double,double>
        surface_deformation_mass_potential_coefficient(const unsigned int degree,
                                                       const unsigned int order) const;

        std::pair<double,double>
        cmb_mass_potential_coefficient(const unsigned int degree,
                                       const unsigned int order) const;

        std::pair<double,double>
        tidal_surface_potential_coefficient(const unsigned int degree,
                                            const unsigned int order) const;

        std::pair<double,double>
        rotational_surface_potential_coefficient(const unsigned int degree,
                                                 const unsigned int order) const;

        Tensor<1,dim>
        reference_frame_body_force(const Point<dim> &position) const;

        double surface_density_jump() const;

        double cmb_density_jump() const;

        bool potential_is_converged() const;

        bool is_self_gravity_active() const
        {
          return self_gravity_active;
        }

        const PotentialFeedback::Settings &
        get_settings() const
        {
          return settings;
        }

        const PotentialFeedback::SelfGravitation<dim> &
        get_self_gravity() const
        {
          return self_gravity;
        }

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

        void save(std::map<std::string, std::string> &status_strings) const override;

        void load(const std::map<std::string, std::string> &status_strings) override;

      private:
        bool mechanism_is_active(const std::string &name) const;

        void set_active_feedback_boundaries_from_traction_model();

        PotentialFeedbackTraction<dim> &
        primary_provider();

        const PotentialFeedbackTraction<dim> &
        primary_provider() const;

        PotentialFeedback::Settings settings;
        bool self_gravity_active = false;
        bool rotational_feedback_active = false;
        bool glacial_isostatic_adjustment_active = false;

        PotentialFeedback::SelfGravitation<dim> self_gravity;
        PotentialFeedback::RotationalFeedback<dim> rotational_feedback;
        PotentialFeedback::GlacialIsostaticAdjustment<dim>
        glacial_isostatic_adjustment;
    };
  }
}

#endif
