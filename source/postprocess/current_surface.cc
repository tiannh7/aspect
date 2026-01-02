/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

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


#include <aspect/postprocess/current_surface.h>
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/spherical_shell.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>

#include <cmath>
#include <limits>


namespace
{
  using namespace dealii;

  /**
   * A simple interpolator for scattered data using a grid-based search structure.
   * It performs Inverse Distance Weighting (IDW) interpolation.
   */
  template <int dim>
  class ScatteredDataInterpolator : public Function<dim>
  {
    public:
      ScatteredDataInterpolator(const std::vector<Point<dim>> &points,
                                const std::vector<double>     &values)
        : Function<dim>(1)
        , points(points)
        , values(values)
      {
        Assert(points.size() == values.size(), ExcDimensionMismatch(points.size(), values.size()));

        if (points.empty())
          return;

        // Determine bounding box
        Point<dim> min_p = points[0];
        Point<dim> max_p = points[0];
        for (const auto &p : points)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                min_p[d] = std::min(min_p[d], p[d]);
                max_p[d] = std::max(max_p[d], p[d]);
              }
          }

        // Expand bounding box slightly to avoid boundary issues
        for (unsigned int d = 0; d < dim; ++d)
          {
            double width = max_p[d] - min_p[d];
            if (width < 1e-10) width = 1.0;
            min_p[d] -= 0.01 * width;
            max_p[d] += 0.01 * width;
            grid_origin[d] = min_p[d];
            grid_width[d] = max_p[d] - min_p[d];
          }

        // Determine grid size. Target roughly 10 points per bin on average?
        // Or just sqrt(N) per dimension.
        unsigned int n_points = points.size();
        unsigned int n_bins_total = std::max(1u, n_points / 4); // 4 points per bin average
        double bin_volume = 1.0;
        for (unsigned int d=0; d<dim; ++d) bin_volume *= grid_width[d];

        double bin_side = std::pow(bin_volume / n_bins_total, 1.0/dim);

        for (unsigned int d = 0; d < dim; ++d)
          {
            grid_size[d] = std::max(1u, static_cast<unsigned int>(grid_width[d] / bin_side));
          }

        // Fill bins
        unsigned int total_bins = 1;
        for (unsigned int d = 0; d < dim; ++d) total_bins *= grid_size[d];
        bins.resize(total_bins);

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            unsigned int bin_idx = get_bin_index(points[i]);
            if (bin_idx < bins.size())
              bins[bin_idx].push_back(i);
          }
      }

      double value(const Point<dim> &p, const unsigned int component = 0) const override
      {
        (void)component;
        if (points.empty()) return 0.0;

        // Find nearest points
        // For simplicity, we search the bin containing p and its neighbors.
        // If no points found, we expand search (not implemented for simplicity, assuming dense enough points).

        // Actually, for robustness, if we don't find points in immediate neighbors, we should probably return 0 or error.
        // But for surface depth, we expect p to be close to surface.

        std::vector<unsigned int> candidates;
        Tensor<1, dim, int> center_idx = get_grid_indices(p);

        // Iterate over 3^dim neighbors
        // This is a bit tedious to write generically for dim, but dim is small (1 or 2).
        // Since this class is used with dim-1, and dim is 2 or 3, so this class uses dim=1 or 2.

        if (dim == 1)
          {
            for (int i = -1; i <= 1; ++i)
              add_bin_candidates(candidates, center_idx[0] + i);
          }
        else if (dim == 2)
          {
            for (int i = -1; i <= 1; ++i)
              for (int j = -1; j <= 1; ++j)
                add_bin_candidates(candidates, center_idx[0] + i, center_idx[1] + j);
          }

        if (candidates.empty())
          {
            // Fallback: linear search (slow but safe)
            // Or just return closest from a random sample?
            // Let's do linear search if local search fails. It shouldn't happen often if grid is good.
            // Actually, let's just search all points if candidates is empty.
            for (unsigned int i=0; i<points.size(); ++i) candidates.push_back(i);
          }

        // IDW
        double sum_w = 0.0;
        double sum_val = 0.0;
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;

        for (unsigned int idx : candidates)
          {
            double dist = p.distance(points[idx]);
            if (dist < 1e-10) return values[idx];

            if (dist < min_dist)
              {
                min_dist = dist;
                closest_idx = idx;
              }

            // Weight = 1/d^2
            double w = 1.0 / (dist * dist);
            sum_w += w;
            sum_val += w * values[idx];
          }

        if (sum_w > 0)
          return sum_val / sum_w;
        else if (closest_idx != -1)
          return values[closest_idx];

        return 0.0;
      }

    private:
      const std::vector<Point<dim>> &points;
      const std::vector<double>     &values;

      Point<dim> grid_origin;
      Tensor<1, dim> grid_width;
      Tensor<1, dim, unsigned int> grid_size;
      std::vector<std::vector<unsigned int>> bins;

      Tensor<1, dim, int> get_grid_indices(const Point<dim> &p) const
      {
        Tensor<1, dim, int> idx;
        for (unsigned int d = 0; d < dim; ++d)
          {
            double f = (p[d] - grid_origin[d]) / grid_width[d] * grid_size[d];
            idx[d] = static_cast<int>(std::floor(f));
          }
        return idx;
      }

      unsigned int get_bin_index(const Point<dim> &p) const
      {
        Tensor<1, dim, int> idx = get_grid_indices(p);
        return get_bin_index_from_indices(idx);
      }

      unsigned int get_bin_index_from_indices(const Tensor<1, dim, int> &idx) const
      {
        // Check bounds
        for (unsigned int d = 0; d < dim; ++d)
          if (idx[d] < 0 || idx[d] >= static_cast<int>(grid_size[d]))
            return std::numeric_limits<unsigned int>::max();

        unsigned int flat_idx = 0;
        unsigned int stride = 1;
        for (unsigned int d = 0; d < dim; ++d)
          {
            flat_idx += idx[d] * stride;
            stride *= grid_size[d];
          }
        return flat_idx;
      }

      void add_bin_candidates(std::vector<unsigned int> &candidates, int i, int j = 0) const
      {
        Tensor<1, dim, int> idx;
        idx[0] = i;
        if (dim > 1) idx[1] = j;

        unsigned int bin_idx = get_bin_index_from_indices(idx);
        if (bin_idx < bins.size())
          {
            candidates.insert(candidates.end(), bins[bin_idx].begin(), bins[bin_idx].end());
          }
      }
  };
}


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    CurrentSurface<dim>::execute (TableHandler &)
    {
      const bool is_box = Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const bool is_spherical = Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model());

      AssertThrow(is_box || is_spherical,
                  ExcMessage("The current surface postprocessor is only implemented with a 2D box or spherical shell model."));

      this->get_computing_timer().enter_subsection("Geometry model surface update");

      // loop over all of the surface cells and save the elevation to a stored value.
      // This needs to be sent to 1 processor, sorted, and broadcast so that every processor knows the entire surface.
      std::vector<std::vector<double>> local_surface_height;

      types::boundary_id relevant_boundary;
      if (is_box)
        relevant_boundary = this->get_geometry_model().translate_symbolic_boundary_name_to_id ("top");
      else
        relevant_boundary = this->get_geometry_model().translate_symbolic_boundary_name_to_id ("outer");

      const QTrapezoid<dim-1> face_corners;
      FEFaceValues<dim> fe_face_values(this->get_mapping(),
                                       this->get_fe(),
                                       face_corners,
                                       update_quadrature_points);

      // Loop over all corners at the surface and save their position.
      // TODO: Update this to work in 3D and spherical geometries
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face_no : cell->face_indices())
            if (cell->face(face_no)->at_boundary())
              if ( cell->face(face_no)->boundary_id() == relevant_boundary)
                {
                  fe_face_values.reinit(cell, face_no);

                  for (unsigned int corner = 0; corner < face_corners.size(); ++corner)
                    {
                      const Point<dim> vertex = fe_face_values.quadrature_point(corner);

                      // We can't push back a point so we convert it into a vector.
                      // This is needed later to keep the vertices together when sorting.
                      std::vector<double> vertex_row;
                      if (is_box)
                        {
                          for (unsigned int i=0; i<dim; ++i)
                            vertex_row.push_back(vertex[i]);
                        }
                      else // is_spherical
                        {
                          if (dim == 2)
                            {
                              // Convert to spherical coordinates (r, phi)
                              // We store phi as x (coordinate), r as y (value)
                              // phi is in [-pi, pi]
                              double r = vertex.norm();
                              double phi = std::atan2(vertex[1], vertex[0]);
                              vertex_row.push_back(phi);
                              vertex_row.push_back(r);
                            }
                          else
                            {
                              // Convert to spherical coordinates (r, phi, theta)
                              // We store phi, theta as coordinates, r as value
                              double r = vertex.norm();
                              double phi = std::atan2(vertex[1], vertex[0]);
                              double theta = std::acos(vertex[2] / r);
                              vertex_row.push_back(phi);
                              vertex_row.push_back(theta);
                              vertex_row.push_back(r);
                            }
                        }

                      local_surface_height.push_back(vertex_row);
                    }
                }

      // Combine all local_surfaces and broadcast back.
      std::vector<std::vector<double>> temp_surface =
        Utilities::MPI::compute_set_union(local_surface_height, this->get_mpi_communicator());

      if (dim == 2)
        {
          // Sort the vector so that it ascends in X.
          std::sort(temp_surface.begin(), temp_surface.end(), [](const std::vector<double> &a, const std::vector<double> &b)
          {
            return a[0] < b[0];
          });

          if (is_spherical && !temp_surface.empty())
            {
              // Add padding to handle periodicity
              std::vector<double> first = temp_surface.front();
              first[0] += 2 * M_PI;
              temp_surface.push_back(first);

              std::vector<double> last = temp_surface[temp_surface.size() - 2]; // -2 because we just pushed one
              last[0] -= 2 * M_PI;
              temp_surface.insert(temp_surface.begin(), last);
            }

          // Define a comparison to remove duplicate surface points.
          const auto compareRows = [](const std::vector<double> &row1, const std::vector<double> &row2)
          {
            return row1 == row2;
          };

          // Remove non-unique rows from the sorted 2D vector
          const auto last = std::unique(temp_surface.begin(), temp_surface.end(), compareRows);
          temp_surface.erase(last, temp_surface.end());

          // Resize data table.
          TableIndices<dim-1> size_idx;
          for (unsigned int d=0; d<dim-1; ++d)
            size_idx[d] = temp_surface.size();

          data_table.TableBase<dim-1,double>::reinit(size_idx);
          TableIndices<dim-1> idx;

          // Fill the data table with the y-values that correspond to the surface.
          // This only works in 2D and will need to be updated for 3D models.
          for (unsigned int x=0; x<data_table.size()[0]; ++x)
            {
              idx[0] = x;
              data_table(idx) = temp_surface[x][1];
            }

          // Fill the coordinates with the x-values used for the data table.
          // This only works in 2D, and will need to be updated for 3D.
          coordinates[0].resize(temp_surface.size());
          for (unsigned int i=0; i<temp_surface.size(); ++i)
            coordinates[0][i] = temp_surface[i][0];

          // Create a surface function for the elevations.
          surface_function = std::make_unique<Functions::InterpolatedTensorProductGridData<dim-1>>(coordinates, data_table);
        }
      else
        {
          // For 3D, use scattered data interpolation
          scattered_coordinates.clear();
          scattered_values.clear();
          scattered_coordinates.reserve(temp_surface.size());
          scattered_values.reserve(temp_surface.size());

          for (const auto &row : temp_surface)
            {
              Point<dim-1> p;
              for (unsigned int d=0; d<dim-1; ++d)
                p[d] = row[d];

              scattered_coordinates.push_back(p);
              scattered_values.push_back(row[dim-1]);
            }

          if (is_spherical)
            {
              // Handle periodicity in phi (index 0)
              // Duplicate points near -pi to +pi and vice versa
              // Threshold: e.g. 0.1 radians
              const double threshold = 0.2;
              std::vector<Point<dim-1>> extra_points;
              std::vector<double> extra_values;

              for (unsigned int i=0; i<scattered_coordinates.size(); ++i)
                {
                  if (scattered_coordinates[i][0] < -M_PI + threshold)
                    {
                      Point<dim-1> p = scattered_coordinates[i];
                      p[0] += 2 * M_PI;
                      extra_points.push_back(p);
                      extra_values.push_back(scattered_values[i]);
                    }
                  else if (scattered_coordinates[i][0] > M_PI - threshold)
                    {
                      Point<dim-1> p = scattered_coordinates[i];
                      p[0] -= 2 * M_PI;
                      extra_points.push_back(p);
                      extra_values.push_back(scattered_values[i]);
                    }
                }

              scattered_coordinates.insert(scattered_coordinates.end(), extra_points.begin(), extra_points.end());
              scattered_values.insert(scattered_values.end(), extra_values.begin(), extra_values.end());
            }

          surface_function = std::make_unique<ScatteredDataInterpolator<dim-1>>(scattered_coordinates, scattered_values);
        }

      this->get_computing_timer().leave_subsection("Geometry model surface update");

      return std::make_pair ("Storing deformed surface topography: ",
                             "Done");
    }



    template <int dim>
    double
    CurrentSurface<dim>::depth_including_mesh_deformation(const Point<dim> &position) const
    {
      const bool is_box = Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const bool is_spherical = Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model());

      AssertThrow(is_box || is_spherical,
                  ExcMessage("Depth with mesh deformation currently only works with a 2D box or spherical shell geometry model."));

      AssertThrow (this->get_parameters().mesh_deformation_enabled,
                   ExcMessage("Depth with mesh deformation must be used with mesh deformation activated."));

      // Convert the point to dim-1, as we aren't interested in the vertical component
      // for the function.
      Point<dim-1> p;
      double vertical_position;

      if (is_box)
        {
          for (unsigned int d=0; d<dim-1; ++d)
            p[d] = position[d];
          vertical_position = position[dim-1];
        }
      else // is_spherical
        {
          if (dim == 2)
            {
              // Convert to spherical coordinates (r, phi)
              // p[0] = phi
              // vertical_position = r
              double r = position.norm();
              double phi = std::atan2(position[1], position[0]);
              p[0] = phi;
              vertical_position = r;
            }
          else
            {
              // Convert to spherical coordinates (r, phi, theta)
              // p[0] = phi, p[1] = theta
              // vertical_position = r
              double r = position.norm();
              double phi = std::atan2(position[1], position[0]);
              double theta = std::acos(position[2] / r);
              p[0] = phi;
              p[1] = theta;
              vertical_position = r;
            }
        }

      double depth_from_surface = surface_function->value(p) - vertical_position;
      return std::max (depth_from_surface, 0.);
    }



    template <int dim>
    template <class Archive>
    void CurrentSurface<dim>::serialize (Archive &ar, const unsigned int)
    {
      ar &coordinates
      & data_table
      & scattered_coordinates
      & scattered_values;
    }



    template <int dim>
    void
    CurrentSurface<dim>::save (std::map<std::string, std::string> &status_strings) const
    {
      // Serialize into a stringstream. Put the following into a code
      // block of its own to ensure the destruction of the 'oa'
      // archive triggers a flush() on the stringstream so we can
      // query the completed string below.
      std::ostringstream os;
      {
        aspect::oarchive oa (os);
        oa << (*this);
      }

      status_strings["CurrentSurface"] = os.str();
    }


    template <int dim>
    void
    CurrentSurface<dim>::load (const std::map<std::string, std::string> &status_strings)
    {
      // see if something was saved
      if (status_strings.find("CurrentSurface") != status_strings.end())
        {
          std::istringstream is (status_strings.find("CurrentSurface")->second);
          aspect::iarchive ia (is);
          ia >> (*this);
        }

      // Recreate function after loading data.
      if (!scattered_coordinates.empty())
        {
          surface_function = std::make_unique<ScatteredDataInterpolator<dim-1>>(scattered_coordinates, scattered_values);
        }
      else if (!coordinates[0].empty())
        {
          surface_function = std::make_unique<Functions::InterpolatedTensorProductGridData<dim-1>>(coordinates, data_table);
        }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CurrentSurface,
                                  "current surface",
                                  "A postprocessor that computes a function "
                                  "of the surface that includes the mesh deformation. "
                                  "This postprocessor has a function that can be called from other "
                                  "plugins to get the depth.")
  }
}
