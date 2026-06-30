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

namespace aspect
{
  namespace BoundaryTraction
  {
    template <int dim>
    Tensor<1,dim>
    PotentialFeedbackTraction<dim>::
    boundary_traction(const types::boundary_id,
                      const Point<dim> &,
                      const Tensor<1,dim> &) const
    {
      AssertThrow(settings.has_active_mechanisms() == false,
                  ExcMessage("The `potential feedback traction' boundary "
                             "adapter is registered and parses the new "
                             "`Potential feedback' parameter hierarchy, but "
                             "provider-backed traction evaluation has not yet "
                             "been connected. Keep using the legacy `self "
                             "gravitation' and `rotational feedback' boundary "
                             "traction plugins for production runs until the "
                             "next migration step is implemented."));

      return Tensor<1,dim>();
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
    }
  }
}

namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(
      PotentialFeedbackTraction,
      "potential feedback traction",
      "Thin boundary traction adapter for potential-feedback-derived normal "
      "traction. This migration-stage plugin declares and parses the shared "
      "``Potential feedback'' and ``Planet model'' parameter hierarchy. The "
      "actual self-gravity and rotational-feedback physics still live in the "
      "legacy boundary traction plugins until the provider manager is wired.")
  }
}
