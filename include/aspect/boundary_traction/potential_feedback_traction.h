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
#include <aspect/boundary_traction/rotational_feedback.h>
#include <aspect/boundary_traction/self_gravitation.h>
#include <aspect/potential_feedback/settings.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace BoundaryTraction
  {
    /**
     * Thin boundary-traction adapter for potential-feedback-derived normal
     * traction.
     *
     * This class establishes the user-facing boundary traction model name
     * `potential feedback traction' and parses the new shared
     * `Potential feedback' hierarchy. The first migration step keeps the
     * existing self-gravity and rotational-feedback physics in their legacy
     * boundary traction plugins; provider-backed traction evaluation will be
     * connected in the next step.
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

        bool potential_is_converged() const;

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

      private:
        bool mechanism_is_active(const std::string &name) const;

        PotentialFeedback::Settings settings;
        bool self_gravity_active = false;
        bool rotational_feedback_active = false;

        SelfGravitation<dim> self_gravity;
        RotationalFeedback<dim> rotational_feedback;
    };
  }
}

#endif
