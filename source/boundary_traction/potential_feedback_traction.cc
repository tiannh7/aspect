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
    PotentialFeedbackTraction<dim>::initialize()
    {
      if (self_gravity_active)
        self_gravity.initialize();

      if (rotational_feedback_active)
        rotational_feedback.initialize();
    }



    template <int dim>
    void
    PotentialFeedbackTraction<dim>::update()
    {
      if (self_gravity_active)
        self_gravity.update();

      if (rotational_feedback_active)
        rotational_feedback.update();
    }



    template <int dim>
    Tensor<1,dim>
    PotentialFeedbackTraction<dim>::
    boundary_traction(const types::boundary_id boundary_indicator,
                      const Point<dim> &position,
                      const Tensor<1,dim> &normal_vector) const
    {
      Tensor<1,dim> traction;

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
    bool
    PotentialFeedbackTraction<dim>::potential_is_converged() const
    {
      return (!self_gravity_active || self_gravity.potential_is_converged())
             &&
             (!rotational_feedback_active
              || rotational_feedback.potential_is_converged());
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
      self_gravity_active = mechanism_is_active("self gravity");
      rotational_feedback_active =
        mechanism_is_active("rotational feedback");

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
      "traction. The model is configured through the shared ``Planet model'' "
      "and ``Potential feedback'' parameter hierarchies and dispatches the "
      "active self-gravity and rotational-feedback mechanisms without "
      "requiring legacy per-plugin parameter blocks.")
  }
}
