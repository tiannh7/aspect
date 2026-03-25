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


#include <aspect/postprocess/mesh_geometry_statistics.h>

#include <deal.II/base/geometry_info.h>

#include <limits>


namespace aspect
{
  namespace Postprocess
  {
    namespace
    {
      template <int dim>
      std::pair<double, double>
      min_and_max_edge_length(const typename DoFHandler<dim>::active_cell_iterator &cell)
      {
        double min_edge_length = std::numeric_limits<double>::max();
        double max_edge_length = 0.0;

        for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line)
          {
            const unsigned int vertex_0 = GeometryInfo<dim>::line_to_cell_vertices(line, 0);
            const unsigned int vertex_1 = GeometryInfo<dim>::line_to_cell_vertices(line, 1);

            const double edge_length = cell->vertex(vertex_0).distance(cell->vertex(vertex_1));

            min_edge_length = std::min(min_edge_length, edge_length);
            max_edge_length = std::max(max_edge_length, edge_length);
          }

        return {min_edge_length, max_edge_length};
      }
    }


    template <int dim>
    std::pair<std::string,std::string>
    MeshGeometryStatistics<dim>::execute (TableHandler &statistics)
    {
      double local_min_diameter = std::numeric_limits<double>::max();
      double local_max_diameter = 0.0;
      double local_sum_diameter = 0.0;
      double local_min_shortest_edge = std::numeric_limits<double>::max();
      double local_max_shortest_edge = 0.0;
      double local_sum_shortest_edge = 0.0;
      double local_min_longest_edge = std::numeric_limits<double>::max();
      double local_max_longest_edge = 0.0;
      double local_sum_longest_edge = 0.0;
      double local_min_aspect_ratio = std::numeric_limits<double>::max();
      double local_max_aspect_ratio = 0.0;
      double local_sum_aspect_ratio = 0.0;
      unsigned int local_cell_count = 0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const double diameter = cell->diameter();
            const auto [shortest_edge, longest_edge] = min_and_max_edge_length<dim>(cell);

            AssertThrow(shortest_edge > 0.0,
                        ExcMessage("Encountered an active cell with zero edge length while "
                                   "computing mesh geometry statistics."));

            const double aspect_ratio = longest_edge / shortest_edge;

            local_min_diameter = std::min(local_min_diameter, diameter);
            local_max_diameter = std::max(local_max_diameter, diameter);
            local_sum_diameter += diameter;

            local_min_shortest_edge = std::min(local_min_shortest_edge, shortest_edge);
            local_max_shortest_edge = std::max(local_max_shortest_edge, shortest_edge);
            local_sum_shortest_edge += shortest_edge;

            local_min_longest_edge = std::min(local_min_longest_edge, longest_edge);
            local_max_longest_edge = std::max(local_max_longest_edge, longest_edge);
            local_sum_longest_edge += longest_edge;

            local_min_aspect_ratio = std::min(local_min_aspect_ratio, aspect_ratio);
            local_max_aspect_ratio = std::max(local_max_aspect_ratio, aspect_ratio);
            local_sum_aspect_ratio += aspect_ratio;

            ++local_cell_count;
          }

      const MPI_Comm mpi_communicator = this->get_mpi_communicator();

      const double global_min_diameter = Utilities::MPI::min(local_min_diameter, mpi_communicator);
      const double global_max_diameter = Utilities::MPI::max(local_max_diameter, mpi_communicator);
      const double global_sum_diameter = Utilities::MPI::sum(local_sum_diameter, mpi_communicator);
      const double global_min_shortest_edge = Utilities::MPI::min(local_min_shortest_edge, mpi_communicator);
      const double global_max_shortest_edge = Utilities::MPI::max(local_max_shortest_edge, mpi_communicator);
      const double global_sum_shortest_edge = Utilities::MPI::sum(local_sum_shortest_edge, mpi_communicator);
      const double global_min_longest_edge = Utilities::MPI::min(local_min_longest_edge, mpi_communicator);
      const double global_max_longest_edge = Utilities::MPI::max(local_max_longest_edge, mpi_communicator);
      const double global_sum_longest_edge = Utilities::MPI::sum(local_sum_longest_edge, mpi_communicator);
      const double global_min_aspect_ratio = Utilities::MPI::min(local_min_aspect_ratio, mpi_communicator);
      const double global_max_aspect_ratio = Utilities::MPI::max(local_max_aspect_ratio, mpi_communicator);
      const double global_sum_aspect_ratio = Utilities::MPI::sum(local_sum_aspect_ratio, mpi_communicator);
      const unsigned int global_cell_count = Utilities::MPI::sum(local_cell_count, mpi_communicator);
      const double global_avg_diameter = global_sum_diameter / global_cell_count;
      const double global_avg_shortest_edge = global_sum_shortest_edge / global_cell_count;
      const double global_avg_longest_edge = global_sum_longest_edge / global_cell_count;
      const double global_avg_aspect_ratio = global_sum_aspect_ratio / global_cell_count;

      const std::vector<std::string> column_names
      = {"Minimal cell diameter (m)",
         "Average cell diameter (m)",
         "Maximal cell diameter (m)",
         "Minimal shortest edge length (m)",
         "Average shortest edge length (m)",
         "Maximal shortest edge length (m)",
         "Minimal longest edge length (m)",
         "Average longest edge length (m)",
         "Maximal longest edge length (m)",
         "Minimal cell aspect ratio",
         "Average cell aspect ratio",
         "Maximal cell aspect ratio"
        };

      statistics.add_value(column_names[0], global_min_diameter);
      statistics.add_value(column_names[1], global_avg_diameter);
      statistics.add_value(column_names[2], global_max_diameter);
      statistics.add_value(column_names[3], global_min_shortest_edge);
      statistics.add_value(column_names[4], global_avg_shortest_edge);
      statistics.add_value(column_names[5], global_max_shortest_edge);
      statistics.add_value(column_names[6], global_min_longest_edge);
      statistics.add_value(column_names[7], global_avg_longest_edge);
      statistics.add_value(column_names[8], global_max_longest_edge);
      statistics.add_value(column_names[9], global_min_aspect_ratio);
      statistics.add_value(column_names[10], global_avg_aspect_ratio);
      statistics.add_value(column_names[11], global_max_aspect_ratio);

      for (const std::string &column_name : column_names)
        {
          statistics.set_precision(column_name, 8);
          statistics.set_scientific(column_name, true);
        }

      std::ostringstream output;
      output.precision(3);
      output << "\n"
             << "          diameter     : " << global_min_diameter
             << " / " << global_avg_diameter
             << " / " << global_max_diameter << " m\n"
             << "          shortest edge: " << global_min_shortest_edge
             << " / " << global_avg_shortest_edge
             << " / " << global_max_shortest_edge << " m\n"
             << "          longest edge : " << global_min_longest_edge
             << " / " << global_avg_longest_edge
             << " / " << global_max_longest_edge << " m\n"
             << "          aspect ratio : " << global_min_aspect_ratio
             << " / " << global_avg_aspect_ratio
             << " / " << global_max_aspect_ratio;

      return {"Mesh geometry statistics (min/avg/max):", output.str()};
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(MeshGeometryStatistics,
                                  "mesh geometry statistics",
                                  "A postprocessor that computes statistics about the geometry "
                                  "of active mesh cells. In particular, it reports the minimal "
                                  "and maximal cell diameter, shortest edge length, and longest "
                                  "edge length, as well as the minimal, average, and maximal "
                                  "cell aspect ratio. The aspect ratio is defined as the ratio "
                                  "between the longest and shortest edge lengths of each cell.")
  }
}
