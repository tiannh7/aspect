/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

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

#ifndef _aspect_termination_criteria_rms_h
#define _aspect_termination_criteria_rms_h

#include <aspect/termination_criteria/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace TerminationCriteria
  {

    /**
     * A class that terminates the simulation when a specified RMS velocity
     * criterion is met, based on a user-defined function expression.
     *
     * @ingroup TerminationCriteria
     */
    template <int dim>
    class RMS : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate this termination criterion.
         *
         * @return Whether to terminate the simulation (true) or continue
         * (false).
         */
        bool
        execute () override;

        /**
         * Declare parameters for this termination criterion.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters for this termination criterion.
         */
        void
        parse_parameters (ParameterHandler &prm) override;

      private:
        /**
         * The function that determines whether to terminate based on
         * RMS velocity and other variables.
         */
        Functions::ParsedFunction<1> function;

        /**
         * The function expression string for output.
         */
        std::string function_expression;

        /**
         * Compute the RMS velocity.
         */
        double compute_rms_velocity() const;
    };

  }
}

#endif
