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
#include <cstdio>
#include <sstream>

namespace aspect
{
  namespace PotentialFeedback
  {
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
      read_schedule();

      reference_field = load_field(stages.front().file_number);
      update();
    }



    template <int dim>
    void
    SurfaceHistory<dim>::read_schedule()
    {
      stages.clear();

      if (configuration.schedule_file_name.empty())
        {
          stages.push_back({0.0, configuration.first_data_file_number});
          return;
        }

      AssertThrow(Utilities::fexists(configuration.schedule_file_name,
                                     this->get_mpi_communicator()),
                  ExcMessage("Surface-history schedule file <"
                             + configuration.schedule_file_name
                             + "> was not found."));

      std::istringstream input(
        Utilities::read_and_distribute_file_content(
          configuration.schedule_file_name,
          this->get_mpi_communicator()));

      std::vector<std::pair<double,int>> entries;
      std::string line;
      while (std::getline(input, line))
        {
          const std::size_t first_non_space = line.find_first_not_of(" \t\r");
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
                  ExcMessage("Surface-history schedule file <"
                             + configuration.schedule_file_name
                             + "> contains no readable entries."));

      if (configuration.schedule_format == "citcomsve stage ages")
        {
          const unsigned int declared_stage_count =
            static_cast<unsigned int>(entries.front().first);
          entries.erase(entries.begin());

          AssertThrow(entries.size() == declared_stage_count,
                      ExcMessage("The CitcomSVE surface-history schedule "
                                 "declares "
                                 + Utilities::int_to_string(declared_stage_count)
                                 + " stages but contains "
                                 + Utilities::int_to_string(
                                   static_cast<unsigned int>(entries.size()))
                                 + " stage rows."));
          AssertThrow(!entries.empty(), ExcInternalError());

          const double first_age_ka = entries.front().first;
          for (unsigned int index = 0; index < entries.size(); ++index)
            {
              const double elapsed_years =
                (entries[index].first - first_age_ka) * 1000.0;
              AssertThrow(elapsed_years >= 0.0,
                          ExcMessage("CitcomSVE stage ages must progress "
                                     "forward from the first schedule age."));
              stages.push_back(
              {
                elapsed_years * year_in_seconds,
                configuration.first_data_file_number
                + static_cast<int>(index)
              });
            }
        }
      else
        {
          for (const auto &entry : entries)
            {
              double time = entry.first;
              if (this->convert_output_to_years())
                time *= year_in_seconds;

              stages.push_back({time, entry.second});
            }
        }

      AssertThrow(stages.front().time == 0.0,
                  ExcMessage("The first surface-history stage must start at "
                             "elapsed model time zero."));
      for (unsigned int index = 1; index < stages.size(); ++index)
        AssertThrow(stages[index].time > stages[index-1].time,
                    ExcMessage("Surface-history stage times must be strictly "
                               "increasing."));
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
                           [](const double time, const Stage &stage)
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
      if (lower_stage_index != current_lower_stage)
        lower_field = load_field(stages[lower_stage_index].file_number);

      if (upper_stage_index == lower_stage_index)
        upper_field.reset();
      else if (upper_stage_index != current_upper_stage)
        upper_field = load_field(stages[upper_stage_index].file_number);

      current_lower_stage = lower_stage_index;
      current_upper_stage = upper_stage_index;
    }



    template <int dim>
    std::unique_ptr<Utilities::StructuredDataLookup<dim-1>>
    SurfaceHistory<dim>::load_field(const int file_number) const
    {
      const std::string filename = create_filename(file_number);
      AssertThrow(Utilities::fexists(filename, this->get_mpi_communicator()),
                  ExcMessage("Surface-history data file <" + filename
                             + "> was not found."));

      auto lookup =
        std::make_unique<Utilities::StructuredDataLookup<dim-1>>(
          1, configuration.scale_factor);
      lookup->load_file(filename, this->get_mpi_communicator());
      return lookup;
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
        prm.declare_entry("Schedule file name", "",
                          Patterns::Anything(),
                          "Schedule file. An empty value selects a single "
                          "static field.");
        prm.declare_entry("Schedule format",
                          "elapsed time and file number",
                          Patterns::Selection("elapsed time and file number|citcomsve stage ages"),
                          "Schedule syntax. Elapsed-time schedules contain "
                          "time and file number columns. CitcomSVE schedules "
                          "contain a stage-count header followed by age in ka "
                          "and stage time-step count; files are numbered "
                          "sequentially.");
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
