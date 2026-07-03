/*
 Copyright (C) 2015 - 2022 by the authors of the ASPECT code.

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


#ifndef _aspect_postprocess_geoid_h
#define _aspect_postprocess_geoid_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/potential_feedback/self_gravitation.h>


namespace aspect
{
  namespace Postprocess
  {
    /**
     * A postprocessor that computes the geoid anomaly at the surface.
     *
     * @ingroup Postprocessing
     */
    template <int dim>
    class Geoid : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate the solution for the geoid in spherical harmonics and then transfer it to grid output.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;

        /**
         * Register with the simulator the other postprocessors that we need
         * (namely: dynamic topography).
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
         * Find if the top or bottom boundaries are free surfaces.
         */
        void initialize() override;

        void initialize_simulator(const Simulator<dim> &simulator) override;

        /**
         * Evaluate the geoid solution at a point. The evaluation point
         * must be outside of the model domain, and it must be called
         * after execute().
         */
        double
        evaluate (const Point<dim> &) const;

        /**
         * Evaluate the gravity anomaly at a point. The evaluation point
         * must be outside of the model domain, and it must be called
         * after execute().
         */
        double
        evaluate_gravity_anomaly (const Point<dim> &) const;

        std::pair<double,double>
        geoid_coefficient (const unsigned int degree,
                           const unsigned int order) const;

        std::pair<double,double>
        surface_topography_contribution_coefficient (const unsigned int degree,
                                                     const unsigned int order) const;

      private:
        /**
         * Parameters to set the maximum and minimum degree when computing geoid from spherical harmonics
         */
        unsigned int max_degree;
        unsigned int min_degree;

        /**
         * A parameter to control whether to output the data in geographical coordinates.
         * If true, output the data in longitudes and latitudes; if false, output data in x y z.
         */
        bool output_in_lat_lon;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the geoid anomaly
         */
        bool output_geoid_anomaly_SH_coes;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the surface topography contribution
         */
        bool output_surface_topo_contribution_SH_coes;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the CMB topography contribution
         */
        bool output_CMB_topo_contribution_SH_coes;

        /**
         * A parameter to control whether to output the spherical harmonic coefficients of the density anomaly
         */
        bool output_density_anomaly_contribution_SH_coes;

        /**
         * A parameter to control whether to output the free-air gravity anomaly
         */
        bool output_gravity_anomaly;

        /**
         * A parameter to control whether this postprocessor writes its own
         * geoid text files. The spherical-harmonic coefficients are always
         * computed and kept in memory for dependent postprocessors.
         */
        bool output_text_files;

        /** Spherically symmetric reference density removed from the volume
         * integral. It has no physical l>0 signal, but integrating it on a
         * discrete mesh otherwise creates spurious harmonic leakage. */
        double reference_density;

        enum class DensityAnomalyMode
        {
          auto_detect,
          always,
          never
        };

        /**
         * Controls whether the volume-density geoid term is evaluated.
         */
        DensityAnomalyMode density_anomaly_mode;

        /**
         * Absolute tolerance used by the automatic density-anomaly detector.
         */
        double density_anomaly_tolerance;

        /**
         * A parameter to control whether to include the surface topography contribution on geoid
         */
        bool include_surface_topo_contribution;

        /**
         * A parameter to control whether to include the CMB topography contribution on geoid
         */
        bool include_CMB_topo_contribution;

        /**
         * A parameter to specify if the top boundary is an active free surface
         */
        bool use_free_surface_topography;

        /**
         * A parameter to specify if the bottom boundary is an active free surface
         */
        bool use_free_CMB_topography;

        /**
         * Output interval control parameters.
         */
        double       time_between_text_output = 0.;
        unsigned int time_steps_between_text_output = 1;
        double       last_text_output_time = -1e20;

        PotentialFeedback::SelfGravitation<dim> self_gravity_helper;

        /**
         * A vector to store the cosine terms of the geoid anomaly spherical harmonic coefficients.
         */
        std::vector<double> geoid_coecos;
        /**
         * A vector to store the sine terms of the geoid anomaly spherical harmonic coefficients.
         */
        std::vector<double> geoid_coesin;

        std::vector<double> surface_topo_contribution_coecos;
        std::vector<double> surface_topo_contribution_coesin;

        /**
         * Surface gravity value used for gravity anomaly calculations.
         */
        double surface_gravity;

        /**
         * Outer radius of the model domain.
         */
        double outer_radius;
    };
  }
}


#endif
