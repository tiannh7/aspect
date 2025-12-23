/*
  Copyright (C) 2011 - 2023 by the authors of the ASPECT code.

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

#ifndef _aspect_postprocess_composition_density_statistics_h
#define _aspect_postprocess_composition_density_statistics_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {
    /**
     * A postprocessor that computes statistics about the density
     * in compositional fields.
     */
    template <int dim>
    class CompositionDensityStatistics : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate the solution to compute statistics about the density
         * in compositional fields.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;

        /**
         * Declare parameters for this postprocessor.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters for this postprocessor.
         */
        void
        parse_parameters (ParameterHandler &prm) override;
    };
  }
}


#endif
