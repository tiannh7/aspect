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

#include <aspect/potential_feedback/surface_history.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/utilities.h>

#include <deal.II/base/patterns.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace aspect
{
  namespace PotentialFeedback
  {
    namespace SurfaceHistoryUtilities
    {
      std::vector<Stage>
      parse_schedule(const std::string &contents,
                     const std::string &schedule_format,
                     const int first_data_file_number,
                     const bool times_are_years)
      {
        std::vector<std::pair<double,int>> entries;
        std::istringstream input(contents);
        std::string line;
        while (std::getline(input, line))
          {
            const std::size_t first_non_space =
              line.find_first_not_of(" \t\r");
            if (first_non_space == std::string::npos
                || line[first_non_space] == '#')
              continue;

            std::istringstream line_stream(line);
            double first_value = 0.0;
            int second_value = 0;
            if (line_stream >> first_value >> second_value)
              entries.emplace_back(first_value, second_value);
          }

        AssertThrow(!entries.empty(),
                    ExcMessage("The surface-history schedule contains no "
                               "readable entries."));

        std::vector<Stage> stages;
        if (schedule_format == "citcomsve stage ages")
          {
            const double declared_stage_count_value = entries.front().first;
            const unsigned int declared_stage_count =
              static_cast<unsigned int>(declared_stage_count_value);
            AssertThrow(declared_stage_count_value > 0.0
                        && declared_stage_count_value
                        == static_cast<double>(declared_stage_count),
                        ExcMessage("The CitcomSVE surface-history stage "
                                   "count must be a positive integer."));
            entries.erase(entries.begin());

            const std::size_t expected_age_rows =
              static_cast<std::size_t>(declared_stage_count) + 1;
            AssertThrow(entries.size() == expected_age_rows,
                        ExcMessage("The CitcomSVE surface-history schedule "
                                   "declares "
                                   + Utilities::int_to_string(
                                     declared_stage_count)
                                   + " stage intervals but contains "
                                   + Utilities::int_to_string(
                                     static_cast<unsigned int>(entries.size()))
                                   + " age rows. Canonical CitcomSVE schedules "
                                   "require one more age row than interval."));
            AssertThrow(!entries.empty(), ExcInternalError());

            const double first_age_ka = entries.front().first;
            const double chronological_direction =
              (first_age_ka > 0.0 ? -1.0 : 1.0);
            for (unsigned int index = 0; index < entries.size(); ++index)
              {
                const double elapsed_years =
                  chronological_direction
                  * (entries[index].first - first_age_ka) * 1000.0;
                AssertThrow(elapsed_years >= 0.0,
                            ExcMessage("CitcomSVE stage ages must progress "
                                       "chronologically from the first "
                                       "schedule age."));
                stages.push_back(
                {
                  elapsed_years * year_in_seconds,
                  first_data_file_number + static_cast<int>(index)
                });
              }
          }
        else
          {
            AssertThrow(schedule_format == "elapsed time and file number",
                        ExcMessage("Unknown surface-history schedule format <"
                                   + schedule_format + ">."));
            for (const auto &entry : entries)
              {
                double time = entry.first;
                if (times_are_years)
                  time *= year_in_seconds;

                stages.push_back({time, entry.second});
              }
          }

        AssertThrow(stages.front().time == 0.0,
                    ExcMessage("The first surface-history stage must start at "
                               "elapsed model time zero."));
        for (unsigned int index = 1; index < stages.size(); ++index)
          AssertThrow(stages[index].time > stages[index-1].time,
                      ExcMessage("Surface-history stage times must be "
                                 "strictly increasing."));

        return stages;
      }



      CitcomSVERegularGrid
      parse_citcomsve_regular_grid(const std::string &contents,
                                   const double scale_factor)
      {
        std::vector<std::vector<double>> rows;
        std::istringstream input(contents);
        std::string line;
        while (std::getline(input, line))
          {
            const std::size_t first_non_space =
              line.find_first_not_of(" \t\r");
            if (first_non_space == std::string::npos
                || line[first_non_space] == '#')
              continue;

            std::istringstream line_stream(line);
            std::vector<double> row;
            double entry = 0.0;
            while (line_stream >> entry)
              row.push_back(entry);
            AssertThrow(line_stream.eof(),
                        ExcMessage("The CitcomSVE regular-grid file contains "
                                   "a non-numeric entry."));
            AssertThrow(!row.empty(), ExcInternalError());
            rows.push_back(std::move(row));
          }

        AssertThrow(!rows.empty(),
                    ExcMessage("The CitcomSVE regular-grid file is empty."));

        unsigned int n_longitudes = 0;
        unsigned int n_latitudes = 0;
        unsigned int first_data_row = 0;
        if (rows.front().size() == 2)
          {
            const double longitude_count = rows.front()[0];
            const double latitude_count = rows.front()[1];
            AssertThrow(longitude_count > 0.0
                        && latitude_count > 0.0
                        && longitude_count
                        == static_cast<double>(
                          static_cast<unsigned int>(longitude_count))
                        && latitude_count
                        == static_cast<double>(
                          static_cast<unsigned int>(latitude_count)),
                        ExcMessage("The CitcomSVE regular-grid dimensions "
                                   "must be positive integers."));
            n_longitudes = static_cast<unsigned int>(longitude_count);
            n_latitudes = static_cast<unsigned int>(latitude_count);
            first_data_row = 1;
          }
        else
          {
            AssertThrow(rows.front().size() == 3,
                        ExcMessage("The CitcomSVE regular-grid file must "
                                   "contain either an `nlon nlat' header or "
                                   "three-column longitude, latitude, value "
                                   "rows."));
            const double first_latitude = rows.front()[1];
            while (n_longitudes < rows.size()
                   && rows[n_longitudes].size() == 3
                   && std::abs(rows[n_longitudes][1] - first_latitude)
                   <= 1e-12)
              ++n_longitudes;
            AssertThrow(n_longitudes > 0
                        && rows.size() % n_longitudes == 0,
                        ExcMessage("Could not infer rectangular dimensions "
                                   "from the headerless CitcomSVE regular "
                                   "grid."));
            n_latitudes =
              static_cast<unsigned int>(rows.size() / n_longitudes);
          }

        AssertThrow(n_longitudes >= 2 && n_latitudes >= 2,
                    ExcMessage("A CitcomSVE regular grid requires at least "
                               "two points in each coordinate direction."));
        AssertThrow(rows.size() - first_data_row
                    == n_longitudes * n_latitudes,
                    ExcMessage("The CitcomSVE regular-grid data row count "
                               "does not match its dimensions."));

        std::vector<double> longitudes(n_longitudes);
        std::vector<double> colatitudes(n_latitudes);
        std::vector<double> values(n_longitudes * n_latitudes);

        for (unsigned int latitude_index = 0;
             latitude_index < n_latitudes;
             ++latitude_index)
          for (unsigned int longitude_index = 0;
               longitude_index < n_longitudes;
               ++longitude_index)
            {
              const auto &row =
                rows[first_data_row + longitude_index
                     + n_longitudes * latitude_index];
              AssertThrow(row.size() == 3,
                          ExcMessage("Every CitcomSVE regular-grid data row "
                                     "must contain longitude, latitude, and "
                                     "one value."));
              const double longitude_degrees = row[0];
              const double latitude_degrees = row[1];
              const double value = row[2];

              const double longitude =
                longitude_degrees * numbers::PI / 180.0;
              const double colatitude =
                (90.0 - latitude_degrees) * numbers::PI / 180.0;

              if (latitude_index == 0)
                longitudes[longitude_index] = longitude;
              else
                AssertThrow(std::abs(longitude
                                     - longitudes[longitude_index])
                            <= 1e-12,
                            ExcMessage("CitcomSVE regular-grid longitudes "
                                       "must repeat identically on every "
                                       "latitude row."));

              if (longitude_index == 0)
                colatitudes[latitude_index] = colatitude;
              else
                AssertThrow(std::abs(colatitude
                                     - colatitudes[latitude_index])
                            <= 1e-12,
                            ExcMessage("CitcomSVE regular-grid latitudes "
                                       "must remain constant within a row."));

              values[longitude_index
                     + n_longitudes * latitude_index] = value * scale_factor;
            }

        const double longitude_spacing = longitudes[1] - longitudes[0];
        const double colatitude_spacing = colatitudes[1] - colatitudes[0];
        AssertThrow(longitude_spacing > 0.0 && colatitude_spacing > 0.0,
                    ExcMessage("CitcomSVE regular-grid coordinates must "
                               "increase in longitude and colatitude."));

        for (unsigned int i = 1; i < n_longitudes; ++i)
          AssertThrow(std::abs((longitudes[i] - longitudes[i-1])
                               - longitude_spacing) <= 1e-12,
                      ExcMessage("CitcomSVE regular-grid longitudes must be "
                                 "uniformly spaced."));
        for (unsigned int i = 1; i < n_latitudes; ++i)
          AssertThrow(std::abs((colatitudes[i] - colatitudes[i-1])
                               - colatitude_spacing) <= 1e-12,
                      ExcMessage("CitcomSVE regular-grid latitudes must be "
                                 "uniformly spaced."));

        AssertThrow(std::abs(longitude_spacing * n_longitudes
                             - 2.0 * numbers::PI) <= 1e-10
                    && std::abs(colatitude_spacing * n_latitudes
                                - numbers::PI) <= 1e-10,
                    ExcMessage("The CitcomSVE regular grid must cover the "
                               "complete sphere."));

        CitcomSVERegularGrid result;
        result.coordinate_values.resize(2);
        result.coordinate_values[0].resize(n_longitudes + 2);
        result.coordinate_values[1].resize(n_latitudes + 2);

        result.coordinate_values[0][0] =
          longitudes.front() - longitude_spacing;
        for (unsigned int i = 0; i < n_longitudes; ++i)
          result.coordinate_values[0][i+1] = longitudes[i];
        result.coordinate_values[0][n_longitudes+1] =
          longitudes.back() + longitude_spacing;

        result.coordinate_values[1][0] = 0.0;
        for (unsigned int i = 0; i < n_latitudes; ++i)
          result.coordinate_values[1][i+1] = colatitudes[i];
        result.coordinate_values[1][n_latitudes+1] = numbers::PI;

        Table<2,double> data_table;
        data_table.TableBase<2,double>::reinit(
          TableIndices<2>(n_longitudes + 2, n_latitudes + 2));

        for (unsigned int latitude_index = 0;
             latitude_index < n_latitudes;
             ++latitude_index)
          {
            for (unsigned int longitude_index = 0;
                 longitude_index < n_longitudes;
                 ++longitude_index)
              data_table[longitude_index+1][latitude_index+1] =
                values[longitude_index
                       + n_longitudes * latitude_index];

            data_table[0][latitude_index+1] =
              values[n_longitudes-1
                     + n_longitudes * latitude_index];
            data_table[n_longitudes+1][latitude_index+1] =
              values[n_longitudes * latitude_index];
          }

        double north_pole_value = 0.0;
        double south_pole_value = 0.0;
        for (unsigned int longitude_index = 0;
             longitude_index < n_longitudes;
             ++longitude_index)
          {
            north_pole_value += values[longitude_index];
            south_pole_value +=
              values[longitude_index
                     + n_longitudes * (n_latitudes-1)];
          }
        north_pole_value /= n_longitudes;
        south_pole_value /= n_longitudes;

        for (unsigned int longitude_index = 0;
             longitude_index < n_longitudes + 2;
             ++longitude_index)
          {
            data_table[longitude_index][0] = north_pole_value;
            data_table[longitude_index][n_latitudes+1] = south_pole_value;
          }

        result.data_tables.push_back(std::move(data_table));
        return result;
      }
    }



    template <int dim>
    void
    SurfaceHistory<dim>::configure(
      const SurfaceHistoryConfiguration &new_configuration)
    {
      configuration = new_configuration;
      configuration.data_directory =
        Utilities::expand_ASPECT_SOURCE_DIR(configuration.data_directory);
      configuration.schedule_file_name =
        Utilities::expand_ASPECT_SOURCE_DIR(configuration.schedule_file_name);

      if (!configuration.data_directory.empty()
          && configuration.data_directory.back() != '/')
        configuration.data_directory += '/';

      if (!configuration.schedule_file_name.empty()
          && configuration.schedule_file_name.front() != '/')
        configuration.schedule_file_name =
          configuration.data_directory + configuration.schedule_file_name;
    }



    template <int dim>
    void
    SurfaceHistory<dim>::initialize(const types::boundary_id boundary_id)
    {
      AssertThrow(dim == 3,
                  ExcMessage("Surface histories for glacial isostatic "
                             "adjustment are implemented only in 3D."));
      AssertThrow(Plugins::plugin_type_matches<
                  const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Surface histories for glacial isostatic "
                             "adjustment require a spherical shell geometry."));
      AssertThrow(!configuration.data_file_name.empty(),
                  ExcMessage("A surface-history data file name is required."));

      surface_boundary_id = boundary_id;
      history_communicator =
        std::make_unique<Utilities::MPI::DuplicatedCommunicator>(
          this->get_mpi_communicator());
      read_schedule();

      reference_field = load_field(stages.front().file_number);
      update();
    }



    template <int dim>
    void
    SurfaceHistory<dim>::read_schedule()
    {
      stages.clear();
      Assert(history_communicator, ExcInternalError());
      const MPI_Comm communicator = **history_communicator;

      if (configuration.schedule_file_name.empty())
        {
          stages.push_back({0.0, configuration.first_data_file_number});
          return;
        }

      AssertThrow(Utilities::fexists(configuration.schedule_file_name,
                                     communicator),
                  ExcMessage("Surface-history schedule file <"
                             + configuration.schedule_file_name
                             + "> was not found."));

      const std::string schedule_contents =
        Utilities::read_and_distribute_file_content(
          configuration.schedule_file_name,
          communicator);
      stages = SurfaceHistoryUtilities::parse_schedule(
                 schedule_contents,
                 configuration.schedule_format,
                 configuration.first_data_file_number,
                 this->convert_output_to_years());
    }



    template <int dim>
    void
    SurfaceHistory<dim>::update()
    {
      AssertThrow(!stages.empty(),
                  ExcMessage("SurfaceHistory::initialize() must be called "
                             "before update()."));

      const double current_time = elapsed_model_time();
      const auto upper = std::upper_bound(
                           stages.begin(),
                           stages.end(),
                           current_time,
                           [](const double time,
                              const SurfaceHistoryUtilities::Stage &stage)
      {
        return time < stage.time;
      });

      unsigned int lower_index = 0;
      unsigned int upper_index = 0;
      if (upper == stages.begin())
        lower_index = upper_index = 0;
      else if (upper == stages.end())
        lower_index = upper_index = stages.size() - 1;
      else
        {
          upper_index = std::distance(stages.begin(), upper);
          lower_index = upper_index - 1;
        }

      Assert(history_communicator, ExcInternalError());
      const MPI_Comm communicator = **history_communicator;
      lower_index = Utilities::MPI::broadcast(communicator, lower_index, 0);
      upper_index = Utilities::MPI::broadcast(communicator, upper_index, 0);

      load_bracketing_fields(lower_index, upper_index);

      if (!configuration.interpolate || lower_index == upper_index)
        interpolation_weight = 0.0;
      else
        interpolation_weight =
          (current_time - stages[lower_index].time)
          / (stages[upper_index].time - stages[lower_index].time);

      interpolation_weight =
        std::max(0.0, std::min(1.0, interpolation_weight));
    }



    template <int dim>
    void
    SurfaceHistory<dim>::load_bracketing_fields(
      const unsigned int lower_stage_index,
      const unsigned int upper_stage_index)
    {
      Assert(history_communicator, ExcInternalError());
      const MPI_Comm communicator = **history_communicator;
      const unsigned int root_process = 0;
      const unsigned int root_lower_stage =
        Utilities::MPI::broadcast(communicator,
                                  current_lower_stage,
                                  root_process);
      const unsigned int root_upper_stage =
        Utilities::MPI::broadcast(communicator,
                                  current_upper_stage,
                                  root_process);

      const bool local_cache_is_consistent =
        (current_lower_stage == root_lower_stage
         && current_upper_stage == root_upper_stage
         && static_cast<bool>(lower_field)
         == (current_lower_stage != numbers::invalid_unsigned_int)
         && static_cast<bool>(upper_field)
         == (current_upper_stage != numbers::invalid_unsigned_int
             && current_upper_stage != current_lower_stage));
      const bool cache_is_consistent =
        Utilities::MPI::min(static_cast<unsigned int>(
                              local_cache_is_consistent),
                            communicator) == 1;

      if (!cache_is_consistent)
        {
          const unsigned int inconsistent_rank_count =
            Utilities::MPI::sum(static_cast<unsigned int>(
                                  !local_cache_is_consistent),
                                communicator);
          if (Utilities::MPI::this_mpi_process(communicator) == root_process)
            std::cerr
                << "Surface-history cache for <"
                << configuration.data_file_name
                << "> was inconsistent on "
                << inconsistent_rank_count
                << " MPI rank(s); reloading the requested bracket."
                << std::endl;

          lower_field = load_field(stages[lower_stage_index].file_number);
          if (upper_stage_index == lower_stage_index)
            upper_field.reset();
          else
            upper_field = load_field(stages[upper_stage_index].file_number);
        }
      else
        {
          unsigned int lower_field_stage = current_lower_stage;
          unsigned int upper_field_stage = current_upper_stage;

          if (lower_stage_index != lower_field_stage)
            {
              if (lower_stage_index == upper_field_stage && upper_field)
                {
                  lower_field.swap(upper_field);
                  std::swap(lower_field_stage, upper_field_stage);
                }
              else
                {
                  lower_field =
                    load_field(stages[lower_stage_index].file_number);
                  lower_field_stage = lower_stage_index;
                }
            }

          if (upper_stage_index == lower_stage_index)
            upper_field.reset();
          else if (upper_stage_index != upper_field_stage)
            upper_field =
              load_field(stages[upper_stage_index].file_number);
        }

      current_lower_stage = lower_stage_index;
      current_upper_stage = upper_stage_index;
    }



    template <int dim>
    std::unique_ptr<Utilities::StructuredDataLookup<dim-1>>
    SurfaceHistory<dim>::load_field(const int file_number) const
    {
      Assert(history_communicator, ExcInternalError());
      const MPI_Comm communicator = **history_communicator;
      const std::string filename = create_filename(file_number);

      if (configuration.data_format == "aspect structured data")
        {
          AssertThrow(Utilities::fexists(filename, communicator),
                      ExcMessage("Surface-history data file <" + filename
                                 + "> was not found."));
          auto lookup =
            std::make_unique<Utilities::StructuredDataLookup<dim-1>>(
              1, configuration.scale_factor);
          lookup->load_file(filename, communicator);
          return lookup;
        }

      AssertThrow(configuration.data_format == "citcomsve regular grid",
                  ExcInternalError());
      AssertThrow(dim == 3,
                  ExcMessage("CitcomSVE regular-grid surface histories are "
                             "implemented only in 3D."));

      if constexpr (dim == 3)
        {
          // Surface histories replace their bracketing fields during a run.
          // Keeping these dynamic tables in MPI shared-memory windows makes
          // destruction and reconstruction part of the collective call
          // sequence. Distribute the file contents from rank zero, but build
          // ordinary process-local lookup tables to keep replacement local.
          SurfaceHistoryUtilities::CitcomSVERegularGrid grid =
            SurfaceHistoryUtilities::parse_citcomsve_regular_grid(
              Utilities::read_and_distribute_file_content(
                filename, communicator),
              configuration.scale_factor);

          auto lookup =
            std::make_unique<Utilities::StructuredDataLookup<dim-1>>(1, 1.0);
          lookup->reinit({"surface value"},
                         std::move(grid.coordinate_values),
                         std::move(grid.data_tables));
          return lookup;
        }

      return nullptr;
    }



    template <int dim>
    std::string
    SurfaceHistory<dim>::create_filename(const int file_number) const
    {
      const std::string pattern =
        configuration.data_directory + configuration.data_file_name;

      if (pattern.find('%') == std::string::npos)
        return pattern;

      const int buffer_size = pattern.size() + 256;
      std::vector<char> filename(buffer_size);
      const int written = std::snprintf(filename.data(),
                                        filename.size(),
                                        pattern.c_str(),
                                        file_number);
      AssertThrow(written >= 0
                  && static_cast<unsigned int>(written) < filename.size(),
                  ExcMessage("Invalid or excessively long surface-history "
                             "data file name pattern <" + pattern + ">."));
      return std::string(filename.data());
    }



    template <int dim>
    Point<dim-1>
    SurfaceHistory<dim>::surface_coordinates(
      const Point<dim> &position) const
    {
      Assert(surface_boundary_id != numbers::invalid_boundary_id,
             ExcInternalError());
      const std::array<double,dim> natural_coordinates =
        this->get_geometry_model().cartesian_to_natural_coordinates(position);

      Point<dim-1> coordinates;
      if constexpr (dim == 3)
        {
          coordinates[0] = natural_coordinates[1];
          coordinates[1] = natural_coordinates[2];
        }
      else
        coordinates[0] = natural_coordinates[1];

      return coordinates;
    }



    template <int dim>
    double
    SurfaceHistory<dim>::value(const Point<dim> &position) const
    {
      AssertThrow(lower_field != nullptr,
                  ExcMessage("SurfaceHistory::initialize() must be called "
                             "before querying values."));
      const Point<dim-1> coordinates = surface_coordinates(position);
      const double lower_value = lower_field->get_data(coordinates, 0);

      if (upper_field == nullptr)
        return lower_value;

      return (1.0 - interpolation_weight) * lower_value
             + interpolation_weight * upper_field->get_data(coordinates, 0);
    }



    template <int dim>
    double
    SurfaceHistory<dim>::initial_value(const Point<dim> &position) const
    {
      AssertThrow(reference_field != nullptr,
                  ExcMessage("SurfaceHistory::initialize() must be called "
                             "before querying reference values."));
      return reference_field->get_data(surface_coordinates(position), 0);
    }



    template <int dim>
    bool
    SurfaceHistory<dim>::initialized() const
    {
      return reference_field != nullptr;
    }



    template <int dim>
    double
    SurfaceHistory<dim>::elapsed_model_time() const
    {
      double current_time = this->get_parameters().start_time;
      if (this->simulator_is_past_initialization())
        current_time = this->get_time();

      return std::max(0.0,
                      current_time - this->get_parameters().start_time);
    }



    template <int dim>
    void
    SurfaceHistory<dim>::declare_parameters(
      ParameterHandler &prm,
      const std::string &subsection_name,
      const std::string &default_directory,
      const std::string &default_file_name)
    {
      prm.enter_subsection(subsection_name);
      {
        prm.declare_entry("Data directory", default_directory,
                          Patterns::DirectoryName(),
                          "Directory containing the structured surface "
                          "history files.");
        prm.declare_entry("Data file name", default_file_name,
                          Patterns::Anything(),
                          "File name or printf-style integer pattern for the "
                          "structured surface fields.");
        prm.declare_entry("Data format", "aspect structured data",
                          Patterns::Selection("aspect structured data|citcomsve regular grid"),
                          "Input format. `aspect structured data' uses the "
                          "standard ASPECT # POINTS format with coordinates "
                          "in radians. `citcomsve regular grid' reads the "
                          "canonical CitcomSVE nlon-by-nlat longitude/latitude "
                          "grid in degrees.");
        prm.declare_entry("Schedule file name", "",
                          Patterns::Anything(),
                          "Schedule file. An empty value selects a single "
                          "static field.");
        prm.declare_entry("Schedule format",
                          "elapsed time and file number",
                          Patterns::Selection("elapsed time and file number|citcomsve stage ages"),
                          "Schedule syntax. Elapsed-time schedules contain "
                          "time and file number columns. CitcomSVE schedules "
                          "contain a stage-interval-count header followed by "
                          "one more row of age in ka and stage time-step count; "
                          "files are numbered sequentially.");
        prm.declare_entry("First data file number", "0",
                          Patterns::Integer(),
                          "File number of the first CitcomSVE stage or of a "
                          "single static field whose name contains an integer "
                          "placeholder.");
        prm.declare_entry("Interpolate between stages", "true",
                          Patterns::Bool(),
                          "Whether to linearly interpolate fields between "
                          "successive schedule stages.");
        prm.declare_entry("Scale factor", "1.0",
                          Patterns::Double(),
                          "Multiplicative scale applied to values read from "
                          "the structured data files.");
      }
      prm.leave_subsection();
    }



    template <int dim>
    SurfaceHistoryConfiguration
    SurfaceHistory<dim>::parse_parameters(
      ParameterHandler &prm,
      const std::string &subsection_name)
    {
      SurfaceHistoryConfiguration parsed_configuration;
      prm.enter_subsection(subsection_name);
      {
        parsed_configuration.data_directory = prm.get("Data directory");
        parsed_configuration.data_file_name = prm.get("Data file name");
        parsed_configuration.data_format = prm.get("Data format");
        parsed_configuration.schedule_file_name =
          prm.get("Schedule file name");
        parsed_configuration.schedule_format = prm.get("Schedule format");
        parsed_configuration.first_data_file_number =
          prm.get_integer("First data file number");
        parsed_configuration.interpolate =
          prm.get_bool("Interpolate between stages");
        parsed_configuration.scale_factor = prm.get_double("Scale factor");
      }
      prm.leave_subsection();
      return parsed_configuration;
    }
  }
}

namespace aspect
{
  namespace PotentialFeedback
  {
    template class SurfaceHistory<2>;
    template class SurfaceHistory<3>;
  }
}
