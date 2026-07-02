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


#ifndef _aspect_postprocess_surface_displacement_spherical_harmonics_h
#define _aspect_postprocess_surface_displacement_spherical_harmonics_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * A postprocessor that integrates tangential surface velocity through time
     * and projects the cumulative tangential displacement onto real spherical
     * harmonics.
     *
     * The projected coefficient is the scalar poloidal displacement amplitude
     * @f$V_{lm}@f$ in @f$u_t = V_{lm} \nabla_s Y_{lm}@f$. Dividing it by the
     * configured load displacement scale gives the horizontal load Love number
     * @f$l_l@f$ for single-harmonic load benchmarks.
     */
    template <int dim>
    class SurfaceDisplacementSphericalHarmonics : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Integrate the current top-boundary tangential velocity and write
         * spherical-harmonic coefficients.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;

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
         * Reference displacement amplitude used to convert displacement
         * coefficients to horizontal Love numbers.
         */
        double load_displacement_scale;

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
