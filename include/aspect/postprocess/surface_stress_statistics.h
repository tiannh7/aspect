/*
  Copyright (C) 2011 - 2025 by the authors of the ASPECT code.

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


#ifndef _aspect_postprocess_surface_stress_statistics_h
#define _aspect_postprocess_surface_stress_statistics_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * A postprocessor that computes min/avg/max statistics of the stress tensor
     * on boundary faces (i.e., surface). The averages are area-weighted.
     * If elasticity is enabled, the deviatoric stress from the material model
     * is used; otherwise it is computed from viscosity and strain rate.
     * Optionally converts the tensor to spherical coordinates if requested
     * by the visualization manager.
     */
    template <int dim>
    class SurfaceStressStatistics : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;
    };
  }
}

#endif
