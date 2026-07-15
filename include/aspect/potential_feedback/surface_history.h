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

#ifndef _aspect_potential_feedback_surface_history_h
#define _aspect_potential_feedback_surface_history_h

#include <aspect/simulator_access.h>
#include <aspect/structured_data.h>

#include <deal.II/base/parameter_handler.h>

#include <memory>
#include <string>
#include <vector>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace SurfaceHistoryUtilities
    {
      /** One time/file entry in a parsed surface history schedule. */
      struct Stage
      {
        double time = 0.0;
        int file_number = 0;
      };

      /**
       * Parse a surface-history schedule without accessing simulator state.
       *
       * For an elapsed-time schedule, @p times_are_years determines whether
       * the first column is converted from years to seconds. For a CitcomSVE
       * stage-age schedule, the header gives the number of stage intervals and
       * is followed by one more age row. Ages may use either negative ka
       * relative to the present (the canonical CitcomSVE files) or positive ka
       * BP. In both cases elapsed time is the chronological age difference
       * from the first stage and data files are numbered sequentially.
       */
      std::vector<Stage>
      parse_schedule(const std::string &contents,
                     const std::string &schedule_format,
                     const int first_data_file_number,
                     const bool times_are_years);

      /**
       * Structured data parsed from a canonical CitcomSVE regular-grid file.
       * Coordinates are longitude and colatitude in radians. The returned
       * grid includes periodic longitude ghosts and constant pole rows so it
       * covers the complete spherical surface.
       */
      struct CitcomSVERegularGrid
      {
        std::vector<std::vector<double>> coordinate_values;
        std::vector<Table<2,double>> data_tables;
      };

      /** Parse the canonical `nlon nlat' CitcomSVE surface-grid format. */
      CitcomSVERegularGrid
      parse_citcomsve_regular_grid(const std::string &contents,
                                   const double scale_factor);
    }



    /**
     * Configuration for a time-dependent structured surface field.
     */
    struct SurfaceHistoryConfiguration
    {
      std::string data_directory;
      std::string data_file_name;
      std::string schedule_file_name;
      std::string schedule_format = "elapsed time and file number";
      std::string data_format = "aspect structured data";
      int first_data_file_number = 0;
      bool interpolate = true;
      double scale_factor = 1.0;
    };



    /**
     * A structured surface field sampled at arbitrary, nonuniform stage times.
     *
     * The loader retains the first field as a reference and keeps only the two
     * files that bracket the current model time. It is intended for GIA ice and
     * ocean histories, whose stage durations are not generally uniform.
     */
    template <int dim>
    class SurfaceHistory : public ::aspect::SimulatorAccess<dim>
    {
      public:
        /** Configure the history before initialize() is called. */
        void configure(const SurfaceHistoryConfiguration &configuration);

        /** Load the schedule, reference field, and initial time bracket. */
        void initialize(const types::boundary_id boundary_id);

        /** Update the two bracketing fields for the current model time. */
        void update();

        /** Return the current interpolated field value at @p position. */
        double value(const Point<dim> &position) const;

        /** Return the first-stage reference field value at @p position. */
        double initial_value(const Point<dim> &position) const;

        /** Return whether the history has been initialized. */
        bool initialized() const;

        /** Declare a reusable parameter subsection for this history. */
        static void declare_parameters(ParameterHandler &prm,
                                       const std::string &subsection_name,
                                       const std::string &default_directory,
                                       const std::string &default_file_name);

        /** Parse a history configuration from a reusable subsection. */
        static SurfaceHistoryConfiguration
        parse_parameters(ParameterHandler &prm,
                         const std::string &subsection_name);

      private:
        void read_schedule();

        void load_bracketing_fields(const unsigned int lower_stage_index,
                                    const unsigned int upper_stage_index);

        std::unique_ptr<Utilities::StructuredDataLookup<dim-1>>
        load_field(const int file_number) const;

        std::string create_filename(const int file_number) const;

        Point<dim-1>
        surface_coordinates(const Point<dim> &position) const;

        double elapsed_model_time() const;

        SurfaceHistoryConfiguration configuration;
        types::boundary_id surface_boundary_id = numbers::invalid_boundary_id;
        std::vector<SurfaceHistoryUtilities::Stage> stages;

        unsigned int current_lower_stage = numbers::invalid_unsigned_int;
        unsigned int current_upper_stage = numbers::invalid_unsigned_int;
        double interpolation_weight = 0.0;

        std::unique_ptr<Utilities::StructuredDataLookup<dim-1>> reference_field;
        std::unique_ptr<Utilities::StructuredDataLookup<dim-1>> lower_field;
        std::unique_ptr<Utilities::StructuredDataLookup<dim-1>> upper_field;
    };
  }
}

#endif
