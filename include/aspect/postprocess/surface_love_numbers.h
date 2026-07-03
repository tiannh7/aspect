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


#ifndef _aspect_postprocess_surface_love_numbers_h
#define _aspect_postprocess_surface_love_numbers_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <list>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * A postprocessor that writes one timestep-consistent set of surface
     * coefficients used to compute load Love numbers in postprocessing.
     * The file combines geoid, surface mass-potential, and cumulative
     * tangential displacement coefficients.
     */
    template <int dim>
    class SurfaceLoveNumbers : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Integrate the current top-boundary tangential velocity and write
         * spherical-harmonic coefficients.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;

        /**
         * The unified Love-number output uses geoid and surface
         * mass-potential coefficients computed by the geoid postprocessor.
         */
        std::list<std::string>
        required_other_postprocessors() const override;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm) override;

        /**
         * Save the cumulative displacement coefficients for checkpoint/restart.
         */
        void save (std::map<std::string, std::string> &status_strings) const override;

        /**
         * Restore the cumulative displacement coefficients from checkpoint/restart.
         */
        void load (const std::map<std::string, std::string> &status_strings) override;

        /**
         * Serialize the contents of this class that are not read from the
         * parameter file.
         */
        template <class Archive>
        void serialize (Archive &ar, const unsigned int version);

      private:
        /**
         * Parameters to set the maximum and minimum degree for the spherical
         * harmonic projection.
         */
        unsigned int max_degree;
        unsigned int min_degree;

        /**
         * Whether to combine the in-memory geoid/topography coefficients with
         * the cumulative tangential displacement coefficients into one output
         * file.
         */
        bool output_coefficients = true;

        /**
         * Surface-load properties read from the spherical harmonic load
         * boundary-traction subsection.
         */
        unsigned int load_degree = 0;
        unsigned int load_order = 0;
        double load_height = 0.0;
        double load_density = 0.0;

        /**
         * Time interval used to convert the timestep-zero instantaneous
         * elastic velocity to an initial tangential displacement.
         */
        double initial_elastic_displacement_time = 0.0;

        /**
         * Output interval control parameters.
         */
        double       time_between_text_output = 0.;
        unsigned int time_steps_between_text_output = 1;
        double       last_text_output_time = -1e20;

        /**
         * Cumulative cosine and sine coefficients of the tangential
         * displacement.
         */
        std::vector<double> displacement_coecos;
        std::vector<double> displacement_coesin;
    };
  }
}

#endif
