/*
 Copyright (C) 2015 - 2024 by the authors of the ASPECT code.

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


#include <aspect/simulator.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/utilities.h>
#include <aspect/postprocess/geoid.h>
#include <aspect/postprocess/dynamic_topography.h>
#include <aspect/potential_feedback/self_gravitation.h>
#include <aspect/boundary_traction/potential_feedback_traction.h>
#include <aspect/postprocess/boundary_densities.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/geometry_model/spherical_shell.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    Geoid<dim>::execute (TableHandler &)
    {
      bool output_needed = false;
      if (time_steps_between_text_output > 0 && this->get_timestep_number() % time_steps_between_text_output == 0)
        output_needed = true;
      if (time_between_text_output > 0 && this->get_time() - last_text_output_time >= time_between_text_output)
        output_needed = true;
      if (this->get_timestep_number() == 0)
        output_needed = true;

      if (last_text_output_time == -1e20)
        {
          last_text_output_time = this->get_time();
          output_needed = true;
        }

      if (!output_needed)
        return std::make_pair("", "");
      // Current geoid code only works for spherical shell geometry.
      AssertThrow (Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model())
                   &&
                   dim == 3,
                   ExcMessage("The geoid postprocessor is currently only implemented for the 3d spherical shell geometry model."));

      if (output_text_files)
        Utilities::create_directory (this->get_output_directory() + "geoid/",
                                     this->get_mpi_communicator(),
                                     /* silent=*/true);

      const GeometryModel::SphericalShell<dim> &geometry_model =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>> (this->get_geometry_model());

      // Get the value of the outer radius and inner radius.
      const double outer_radius = geometry_model.outer_radius();
      const double inner_radius = geometry_model.inner_radius();

      const types::boundary_id top_boundary_id = geometry_model.translate_symbolic_boundary_name_to_id("top");

      // Get the value of the surface gravity acceleration from the gravity model.
      Point<dim> surface_point;
      surface_point[0] = outer_radius;
      surface_gravity = this->get_gravity_model().gravity_vector(surface_point).norm();
      this->outer_radius = outer_radius;

      // Get the value of the universal gravitational constant.
      const double G = aspect::constants::big_g;

      // Retrieve active SelfGravitation plugin or fall back to self_gravity_helper
      const auto &traction_manager = this->get_boundary_traction_manager();
      const bool use_self_gravity =
        traction_manager.template has_matching_active_plugin<
        PotentialFeedback::SelfGravitation<dim>>();
      const bool use_potential_feedback =
        traction_manager.template has_matching_active_plugin<
        BoundaryTraction::PotentialFeedbackTraction<dim>>();

      const PotentialFeedback::SelfGravitation<dim> *self_gravity = nullptr;
      if (use_self_gravity)
        {
          self_gravity = &traction_manager.template get_matching_active_plugin<
                         PotentialFeedback::SelfGravitation<dim>>();
        }
      else if (use_potential_feedback)
        {
          const auto &pf = traction_manager.template get_matching_active_plugin<
                           BoundaryTraction::PotentialFeedbackTraction<dim>>();
          if (pf.has_self_gravity_feedback())
            self_gravity = &pf.get_self_gravity();
        }

      // If surface/CMB topography is needed but self-gravity is not active in solver, throw.
      if ((include_surface_topo_contribution || include_CMB_topo_contribution) &&
          self_gravity == nullptr)
        {
          AssertThrow(false,
                      ExcMessage("The geoid postprocessor treats interface densities as physical model parameters, not output parameters. If surface or CMB topography contributions are enabled, activate a `self gravitation` or `potential feedback` boundary traction plugin and define densities in `Potential feedback/Interface properties`."));
        }

      if (self_gravity == nullptr)
        {
          self_gravity = &self_gravity_helper;
        }

      // Compute density anomalies contribution using the helper
      std::pair<std::vector<double>,std::vector<double>> SH_density_coes =
        self_gravity->compute_internal_density_potential(outer_radius);

      std::pair<double, std::pair<std::vector<double>,std::vector<double>>> SH_surface_topo_coes;
      std::pair<double, std::pair<std::vector<double>,std::vector<double>>> SH_CMB_topo_coes;

      double surface_delta_rho = numbers::signaling_nan<double>();
      double CMB_delta_rho = numbers::signaling_nan<double>();

      if (include_surface_topo_contribution == true ||
          include_CMB_topo_contribution == true)
        {
          surface_delta_rho = self_gravity->surface_density_jump();
          CMB_delta_rho = self_gravity->cmb_density_jump();

          // Compute topography potential coefficients using the helper if self-gravity is not active in solver
          if (!use_self_gravity && !use_potential_feedback)
            {
              const auto SH_topo_coes =
                self_gravity->compute_topography_potential(outer_radius, inner_radius);
              SH_surface_topo_coes = SH_topo_coes.first;
              SH_CMB_topo_coes = SH_topo_coes.second;
            }
        }
      // Compute the spherical harmonic coefficients of geoid anomaly.
      std::vector<double> density_anomaly_contribution_coecos;
      std::vector<double> density_anomaly_contribution_coesin;
      std::vector<double> CMB_topo_contribution_coecos;
      std::vector<double> CMB_topo_contribution_coesin;
      std::vector<double> tidal_potential_contribution_coecos;
      std::vector<double> tidal_potential_contribution_coesin;
      geoid_coecos.clear();
      geoid_coesin.clear();
      surface_topo_contribution_coecos.clear();
      surface_topo_contribution_coesin.clear();

      // First compute the spherical harmonic contributions from density anomaly, surface topography and CMB topography.
      int ind = 0; // coefficients index
      for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
        {
          for (unsigned int iord = 0; iord < ideg+1; ++iord)
            {
              double coecos_density_anomaly = (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1))) * SH_density_coes.first.at(ind);
              double coesin_density_anomaly = (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1))) * SH_density_coes.second.at(ind);
              density_anomaly_contribution_coecos.push_back(coecos_density_anomaly);
              density_anomaly_contribution_coesin.push_back(coesin_density_anomaly);

              const bool use_boundary_potential =
                use_self_gravity || use_potential_feedback;

              if (include_surface_topo_contribution == true || include_CMB_topo_contribution == true)
                {
                  const std::pair<double,double> self_gravity_surface =
                    (use_boundary_potential
                     ? self_gravity->surface_mass_potential_coefficient(ideg, iord)
                     : std::pair<double,double> {0.0, 0.0});
                  const double coecos_surface_topo =
                    (use_boundary_potential
                     ? self_gravity_surface.first
                     : (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1)))
                     * surface_delta_rho*SH_surface_topo_coes.second.first.at(ind)*outer_radius);
                  const double coesin_surface_topo =
                    (use_boundary_potential
                     ? self_gravity_surface.second
                     : (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1)))
                     * surface_delta_rho*SH_surface_topo_coes.second.second.at(ind)*outer_radius);
                  surface_topo_contribution_coecos.push_back(coecos_surface_topo);
                  surface_topo_contribution_coesin.push_back(coesin_surface_topo);

                  const std::pair<double,double> self_gravity_cmb =
                    (use_boundary_potential
                     ? self_gravity->cmb_mass_potential_coefficient(ideg, iord)
                     : std::pair<double,double> {0.0, 0.0});
#if DEAL_II_VERSION_GTE(9,6,0)
                  const double coecos_CMB_topo =
                    (use_boundary_potential
                     ? self_gravity_cmb.first
                     : (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1)))
                     * CMB_delta_rho*SH_CMB_topo_coes.second.first.at(ind)*inner_radius*Utilities::pow(inner_radius/outer_radius,ideg+1));
                  const double coesin_CMB_topo =
                    (use_boundary_potential
                     ? self_gravity_cmb.second
                     : (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1)))
                     * CMB_delta_rho*SH_CMB_topo_coes.second.second.at(ind)*inner_radius*Utilities::pow(inner_radius/outer_radius,ideg+1));
#else
                  const double coecos_CMB_topo =
                    (use_boundary_potential
                     ? self_gravity_cmb.first
                     : (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1)))
                     * CMB_delta_rho*SH_CMB_topo_coes.second.first.at(ind)*inner_radius*std::pow(inner_radius/outer_radius,ideg+1));
                  const double coesin_CMB_topo =
                    (use_boundary_potential
                     ? self_gravity_cmb.second
                     : (4 * numbers::PI * G / (surface_gravity * (2 * ideg + 1)))
                     * CMB_delta_rho*SH_CMB_topo_coes.second.second.at(ind)*inner_radius*std::pow(inner_radius/outer_radius,ideg+1));
#endif
                  CMB_topo_contribution_coecos.push_back(coecos_CMB_topo);
                  CMB_topo_contribution_coesin.push_back(coesin_CMB_topo);

                }

              const std::pair<double,double> tidal_potential =
                (use_boundary_potential
                 ? self_gravity->tidal_surface_potential_coefficient(ideg, iord)
                 : std::pair<double,double> {0.0, 0.0});
              tidal_potential_contribution_coecos.push_back(tidal_potential.first);
              tidal_potential_contribution_coesin.push_back(tidal_potential.second);

              ++ind;
            }
        }

      // Then sum the three contributions together to get the spherical harmonic coefficients of geoid anomaly.
      ind = 0; // coefficients index
      for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
        {
          for (unsigned int iord = 0; iord < ideg+1; ++iord)
            {
              if (include_surface_topo_contribution == true || include_CMB_topo_contribution == true)
                {
                  geoid_coecos.push_back(density_anomaly_contribution_coecos.at(ind)
                                         + surface_topo_contribution_coecos.at(ind)
                                         + CMB_topo_contribution_coecos.at(ind)
                                         + tidal_potential_contribution_coecos.at(ind));
                  geoid_coesin.push_back(density_anomaly_contribution_coesin.at(ind)
                                         + surface_topo_contribution_coesin.at(ind)
                                         + CMB_topo_contribution_coesin.at(ind)
                                         + tidal_potential_contribution_coesin.at(ind));
                }
              else
                {
                  geoid_coecos.push_back(density_anomaly_contribution_coecos.at(ind)
                                         + tidal_potential_contribution_coecos.at(ind));
                  geoid_coesin.push_back(density_anomaly_contribution_coesin.at(ind)
                                         + tidal_potential_contribution_coesin.at(ind));
                }

              ind += 1;
            }
        }

      const QMidpoint<dim-1> quadrature_formula_face_center;
      Assert(quadrature_formula_face_center.size() == 1, ExcInternalError());
      FEFaceValues<dim> fe_face_center_values (this->get_mapping(),
                                               this->get_fe(),
                                               quadrature_formula_face_center,
                                               update_values |
                                               update_quadrature_points|
                                               update_JxW_values);

      // Define a vector to store the location of the cells along the surface.
      std::vector<Point<dim>> surface_cell_locations;

      // Loop over all the cells to get the locations of the surface cells to prepare for the geoid computation.
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            // If the cell is at the top boundary, store the cell's upper face midpoint location.
            for (const unsigned int f : cell->face_indices())
              if (cell->at_boundary(f) && cell->face(f)->boundary_id() == top_boundary_id)
                {
                  fe_face_center_values.reinit(cell,f);
                  const Point<dim> midpoint_at_top_face = fe_face_center_values.get_quadrature_points().at(0);
                  surface_cell_locations.push_back(midpoint_at_top_face);
                  break;
                }
          }

      // Transfer the geocentric coordinates of the surface cells to the surface spherical coordinates (theta,phi)
      std::vector<std::pair<double,double>> surface_cell_spherical_coordinates;
      for (unsigned int i=0; i<surface_cell_locations.size(); ++i)
        {
          const std::array<double,dim> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(surface_cell_locations.at(i));
          const double phi = scoord[1];
          const double theta = scoord[2];
          surface_cell_spherical_coordinates.emplace_back(theta,phi);
        }

      // Compute the grid geoid anomaly based on spherical harmonics.
      std::vector<double> geoid_anomaly;
      for (const auto &surface_cell_spherical_coordinate : surface_cell_spherical_coordinates)
        {
          int ind = 0;
          double geoid_value = 0;
          for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
            {
              for (unsigned int iord = 0; iord < ideg+1; ++iord)
                {
                  // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                  const std::pair<double,double> sph_harm_vals = aspect::Utilities::real_spherical_harmonic(ideg,iord,surface_cell_spherical_coordinate.first,surface_cell_spherical_coordinate.second);
                  const double cos_component = sph_harm_vals.first; // real / cos part
                  const double sin_component = sph_harm_vals.second; // imaginary / sin part

                  geoid_value += geoid_coecos.at(ind)*cos_component+geoid_coesin.at(ind)*sin_component;
                  ++ind;
                }
            }
          geoid_anomaly.push_back(geoid_value);
        }

      // The user can get the spherical harmonic coefficients of the density anomaly contribution if needed
      if (output_text_files && output_density_anomaly_contribution_SH_coes == true)
        {
          // Have a stream into which we write the SH coefficients data from density anomaly contribution.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_density_anomaly_contribution_SH_coes;

          // Prepare the output SH coefficients data from density anomaly contribution.
          unsigned int SH_coes_ind = 0;
          for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
            {
              for (unsigned int iord = 0; iord < ideg+1; ++iord)
                {
                  output_density_anomaly_contribution_SH_coes << ideg
                                                              << ' '
                                                              << iord
                                                              << ' '
                                                              << density_anomaly_contribution_coecos.at(SH_coes_ind)
                                                              << ' '
                                                              << density_anomaly_contribution_coesin.at(SH_coes_ind)
                                                              << std::endl;
                  ++SH_coes_ind;
                }
            }

          const std::string density_anomaly_contribution_SH_coes_filename = this->get_output_directory() +
                                                                            "geoid/density_anomaly_contribution_SH_coefficients." +
                                                                            dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the SH coefficients from density anomaly contribution, we only need to stop by the processor 0 to get the data.
          // On processor 0, collect all the data and put them into the output density anomaly contribution SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream density_anomaly_contribution_SH_coes_file (density_anomaly_contribution_SH_coes_filename);
              density_anomaly_contribution_SH_coes_file << "# "
                                                        << "degree order cosine_coefficient sine_coefficient"
                                                        << std::endl;

              // Write out the data on processor 0.
              density_anomaly_contribution_SH_coes_file << output_density_anomaly_contribution_SH_coes.str();
            }
        }

      // The user can get the spherical harmonic coefficients of the surface topography contribution if needed
      if (output_text_files && output_surface_topo_contribution_SH_coes == true)
        {
          // Have a stream into which we write the SH coefficients data from surface topography contribution.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_surface_topo_contribution_SH_coes;

          // Prepare the output SH coefficients data from surface topography contribution.
          unsigned int SH_coes_ind = 0;
          for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
            {
              for (unsigned int iord = 0; iord < ideg+1; ++iord)
                {
                  output_surface_topo_contribution_SH_coes << ideg
                                                           << ' '
                                                           << iord
                                                           << ' '
                                                           << surface_topo_contribution_coecos.at(SH_coes_ind)
                                                           << ' '
                                                           << surface_topo_contribution_coesin.at(SH_coes_ind)
                                                           << std::endl;
                  ++SH_coes_ind;
                }
            }

          const std::string surface_topo_contribution_SH_coes_filename = this->get_output_directory() +
                                                                         "geoid/surface_topography_contribution_SH_coefficients." +
                                                                         dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the SH coefficients from surface topography contribution,
          // we only need to stop by the processor 0 to get the data. On processor 0, collect all the data
          // and put them into the output surface topography contribution SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream surface_topo_contribution_SH_coes_file (surface_topo_contribution_SH_coes_filename);
              surface_topo_contribution_SH_coes_file << "# "
                                                     << "degree order cosine_coefficient sine_coefficient"
                                                     << std::endl;
              std::ostringstream output_surface_delta_rho;
              output_surface_delta_rho << surface_delta_rho;
              surface_topo_contribution_SH_coes_file << "surface density contrast(kg/m^3): "
                                                     << output_surface_delta_rho.str()
                                                     << std::endl;
              // Write out the data on processor 0
              surface_topo_contribution_SH_coes_file << output_surface_topo_contribution_SH_coes.str();
            }
        }

      // The user can get the spherical harmonic coefficients of the CMB topography contribution if needed.
      if (output_text_files && output_CMB_topo_contribution_SH_coes == true)
        {
          // Have a stream into which we write the SH coefficients data from CMB topography contribution.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_CMB_topo_contribution_SH_coes;

          // Prepare the output SH coefficients data from CMB topography contribution.
          unsigned int SH_coes_ind = 0;
          for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
            {
              for (unsigned int iord = 0; iord < ideg+1; ++iord)
                {
                  output_CMB_topo_contribution_SH_coes << ideg
                                                       << ' '
                                                       << iord
                                                       << ' '
                                                       << CMB_topo_contribution_coecos.at(SH_coes_ind)
                                                       << ' '
                                                       << CMB_topo_contribution_coesin.at(SH_coes_ind)
                                                       << std::endl;
                  ++SH_coes_ind;
                }
            }

          const std::string CMB_topo_contribution_SH_coes_filename = this->get_output_directory() +
                                                                     "geoid/CMB_topography_contribution_SH_coefficients." +
                                                                     dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the SH coefficients from CMB topography contribution, we only need to stop by the processor 0
          // to get the data. On processor 0, collect all the data and put them into the output CMB topography contribution SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream CMB_topo_contribution_SH_coes_file (CMB_topo_contribution_SH_coes_filename);
              CMB_topo_contribution_SH_coes_file << "# "
                                                 << "degree order cosine_coefficient sine_coefficient"
                                                 << std::endl;
              std::ostringstream output_CMB_delta_rho;
              output_CMB_delta_rho << CMB_delta_rho;
              CMB_topo_contribution_SH_coes_file << "CMB density contrast(kg/m^3): "
                                                 << output_CMB_delta_rho.str()
                                                 << std::endl;
              // Write out the data on processor 0
              CMB_topo_contribution_SH_coes_file << output_CMB_topo_contribution_SH_coes.str();
            }
        }

      // The user can get the spherical harmonic coefficients of the geoid anomaly if needed.
      if (output_text_files && output_geoid_anomaly_SH_coes == true)
        {
          // Have a stream into which we write the geoid anomaly SH coefficients data.
          // The text stream is then later sent to processor 0.
          std::ostringstream output_geoid_anomaly_SH_coes;

          // Prepare the output geoid anomaly SH coefficients data.
          unsigned int SH_coes_ind = 0;
          for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
            {
              for (unsigned int iord = 0; iord < ideg+1; ++iord)
                {
                  output_geoid_anomaly_SH_coes << ideg
                                               << ' '
                                               << iord
                                               << ' '
                                               << geoid_coecos.at(SH_coes_ind)
                                               << ' '
                                               << geoid_coesin.at(SH_coes_ind)
                                               << std::endl;
                  ++SH_coes_ind;
                }
            }

          const std::string geoid_anomaly_SH_coes_filename = this->get_output_directory() +
                                                             "geoid/geoid_anomaly_SH_coefficients." +
                                                             dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          // Because each processor already held all the geoid anomaly SH coefficients, we only need to stop by the processor 0 to get the data.
          // On processor 0, collect all the data and put them into the output geoid anomaly SH coefficients file.
          if (dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream geoid_anomaly_SH_coes_file (geoid_anomaly_SH_coes_filename);
              geoid_anomaly_SH_coes_file << "# "
                                         << "degree order cosine_coefficient sine_coefficient"
                                         << std::endl;

              // Write out the data on processor 0.
              geoid_anomaly_SH_coes_file << output_geoid_anomaly_SH_coes.str();
            }
        }

      std::string filename;
      if (output_text_files)
        {
          // Have a stream into which we write the geoid height data. the text stream is then
          // later sent to processor 0.
          std::ostringstream output;

          // On processor 0, write the header lines
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              output << "# "
                     << ((output_in_lat_lon == true)? "longitude latitude" : "x y z")
                     << " geoid_anomaly" << std::endl;
            }

          // Prepare the output data.
          if (output_in_lat_lon == true)
            {
              double lon, lat;
              for (unsigned int i=0; i<surface_cell_spherical_coordinates.size(); ++i)
                {
                  // Transfer the spherical coordinates to geographical coordinates.
                  lat = 90. - surface_cell_spherical_coordinates.at(i).first * constants::radians_to_degree;
                  lon = (surface_cell_spherical_coordinates.at(i).second <= numbers::PI
                         ?
                         surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree
                         :
                         surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree - 360.);

                  // Write the solution to the stream output.
                  output << lon
                         << ' '
                         << lat
                         << ' '
                         << geoid_anomaly.at(i)
                         << std::endl;
                }
            }
          else
            {
              for (unsigned int i=0; i<surface_cell_locations.size(); ++i)
                {
                  // Write the solution to the stream output.
                  output << surface_cell_locations.at(i)
                         << ' '
                         << geoid_anomaly.at(i)
                         << std::endl;
                }
            }

          filename = this->get_output_directory() +
                     "geoid/geoid_anomaly." +
                     dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          Utilities::collect_and_write_file_content(filename, output.str(), this->get_mpi_communicator());
        }

      // Prepare the free-air gravity anomaly output.
      if (output_text_files && output_gravity_anomaly == true)
        {
          // Have a stream into which we write the gravity anomaly data. the text stream is then
          // later sent to processor 0.
          std::ostringstream output_gravity_anomaly;

          // On processor 0, write the header lines:
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              output_gravity_anomaly << "# "
                                     << ((output_in_lat_lon == true)? "longitude latitude" : "x y z")
                                     << " gravity_anomaly" << std::endl;
            }

          // Compute the grid gravity anomaly based on spherical harmonics.
          std::vector<double> gravity_anomaly;
          gravity_anomaly.reserve(surface_cell_spherical_coordinates.size());

          for (const auto &surface_cell_spherical_coordinate : surface_cell_spherical_coordinates)
            {
              int ind = 0;
              double gravity_value = 0;
              for (unsigned int ideg =  min_degree; ideg < max_degree+1; ++ideg)
                {
                  for (unsigned int iord = 0; iord < ideg+1; ++iord)
                    {
                      // Normalization after Dahlen and Tromp (1986) Appendix B.6.
                      const std::pair<double,double> sph_harm_vals = aspect::Utilities::real_spherical_harmonic(ideg,iord,surface_cell_spherical_coordinate.first,surface_cell_spherical_coordinate.second);
                      const double cos_component = sph_harm_vals.first; // real / cos part
                      const double sin_component = sph_harm_vals.second; // imaginary / sin part

                      // The conversion from geoid to gravity anomaly is given by gravity_anomaly = (l-1)*g/R_surface * geoid_anomaly
                      // based on Forte (2007) equation [97].
                      gravity_value += (geoid_coecos.at(ind)*cos_component+geoid_coesin.at(ind)*sin_component) * (ideg - 1) * surface_gravity / outer_radius;
                      ++ind;
                    }
                }
              gravity_anomaly.push_back(gravity_value);
            }

          // Prepare the output data.
          if (output_in_lat_lon == true)
            {
              double lon, lat;
              for (unsigned int i=0; i<surface_cell_spherical_coordinates.size(); ++i)
                {
                  // Transfer the spherical coordinates to geographical coordinates.
                  lat = 90. - surface_cell_spherical_coordinates.at(i).first * constants::radians_to_degree;
                  lon = (surface_cell_spherical_coordinates.at(i).second <= numbers::PI
                         ?
                         surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree
                         :
                         surface_cell_spherical_coordinates.at(i).second * constants::radians_to_degree - 360.);

                  // Write the solution to the stream output.
                  output_gravity_anomaly << lon
                                         << ' '
                                         << lat
                                         << ' '
                                         << gravity_anomaly.at(i)
                                         << std::endl;
                }
            }
          else
            {
              for (unsigned int i=0; i<surface_cell_locations.size(); ++i)
                {
                  // Write the solution to the stream output.
                  output_gravity_anomaly << surface_cell_locations.at(i)
                                         << ' '
                                         << gravity_anomaly.at(i)
                                         << std::endl;
                }
            }

          const std::string filename = this->get_output_directory() +
                                       "geoid/gravity_anomaly." +
                                       dealii::Utilities::int_to_string(this->get_timestep_number(), 5);

          Utilities::collect_and_write_file_content(filename, output_gravity_anomaly.str(), this->get_mpi_communicator());
        }

      last_text_output_time = this->get_time();

      if (output_text_files)
        return std::pair<std::string,std::string>("Writing geoid anomaly:",
                                                  filename);
      else
        return std::pair<std::string,std::string>("Computing geoid coefficients:",
                                                  "in memory");
    }

    template <int dim>
    std::list<std::string>
    Geoid<dim>::required_other_postprocessors() const
    {
      std::list<std::string> deps;

      if ( (include_surface_topo_contribution == true && use_free_surface_topography == false) || (include_CMB_topo_contribution == true && use_free_CMB_topography == false) )
        deps.emplace_back("dynamic topography");

      deps.emplace_back("boundary densities");

      return deps;
    }

    template <int dim>
    double
    Geoid<dim>::evaluate (const Point<dim> &/*p*/) const
    {
      Assert(false, ExcNotImplemented());
      return 0;
    }

    template <>
    double
    Geoid<3>::evaluate (const Point<3> &p) const
    {
      const std::array<double,3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(p);
      const double theta = scoord[2];
      const double phi = scoord[1];
      double value = 0.;

      for (unsigned int ideg=min_degree, k=0; ideg < max_degree+1; ++ideg)
        for (unsigned int iord = 0; iord < ideg+1; ++iord, ++k)
          {
            std::pair<double,double> val = aspect::Utilities::real_spherical_harmonic( ideg, iord, theta, phi );

            value += geoid_coecos[k] * val.first +
                     geoid_coesin[k] * val.second;

          }
      return value;
    }

    template <int dim>
    double
    Geoid<dim>::evaluate_gravity_anomaly (const Point<dim> &/*p*/) const
    {
      Assert(false, ExcNotImplemented());
      return 0;
    }

    template <>
    double
    Geoid<3>::evaluate_gravity_anomaly (const Point<3> &p) const
    {
      const std::array<double,3> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(p);
      const double theta = scoord[2];
      const double phi = scoord[1];
      double value = 0.;

      for (unsigned int ideg=min_degree, k=0; ideg < max_degree+1; ++ideg)
        for (unsigned int iord = 0; iord < ideg+1; ++iord, ++k)
          {
            std::pair<double,double> val = aspect::Utilities::real_spherical_harmonic( ideg, iord, theta, phi );

            // The conversion from geoid to gravity anomaly is given by gravity_anomaly = (l-1)*g/R_surface * geoid_anomaly
            // based on Forte (2007) equation [97].
            value += (geoid_coecos[k] * val.first + geoid_coesin[k] * val.second) * (ideg - 1) * surface_gravity / outer_radius;
          }
      return value;
    }



    namespace
    {
      unsigned int
      geoid_coefficient_index (const unsigned int min_degree,
                               const unsigned int max_degree,
                               const unsigned int degree,
                               const unsigned int order)
      {
        AssertThrow(degree >= min_degree && degree <= max_degree,
                    ExcMessage("Requested a geoid coefficient outside the configured degree range."));
        AssertThrow(order <= degree,
                    ExcMessage("Requested a geoid coefficient with order larger than degree."));

        unsigned int index = 0;
        for (unsigned int current_degree = min_degree; current_degree < degree; ++current_degree)
          index += current_degree + 1;
        index += order;

        return index;
      }
    }



    template <int dim>
    std::pair<double,double>
    Geoid<dim>::geoid_coefficient (const unsigned int degree,
                                   const unsigned int order) const
    {
      const unsigned int index =
        geoid_coefficient_index(min_degree, max_degree, degree, order);

      AssertThrow(index < geoid_coecos.size() && index < geoid_coesin.size(),
                  ExcMessage("Geoid coefficients are not available. Make sure the geoid postprocessor has executed before requesting them."));

      return std::make_pair(geoid_coecos[index], geoid_coesin[index]);
    }



    template <int dim>
    std::pair<double,double>
    Geoid<dim>::surface_topography_contribution_coefficient (const unsigned int degree,
                                                             const unsigned int order) const
    {
      const unsigned int index =
        geoid_coefficient_index(min_degree, max_degree, degree, order);

      if (surface_topo_contribution_coecos.empty() &&
          surface_topo_contribution_coesin.empty())
        return std::make_pair(0.0, 0.0);

      AssertThrow(index < surface_topo_contribution_coecos.size()
                  && index < surface_topo_contribution_coesin.size(),
                  ExcMessage("Surface-topography geoid coefficients are not available. Make sure the geoid postprocessor has executed before requesting them."));

      return std::make_pair(surface_topo_contribution_coecos[index],
                            surface_topo_contribution_coesin[index]);
    }



    template <int dim>
    void
    Geoid<dim>::declare_parameters (ParameterHandler &prm)
    {
      PotentialFeedback::SelfGravitation<dim>::declare_parameters(prm);
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Geoid");
        {
          prm.declare_entry("Include surface topography contribution", "true",
                            Patterns::Bool(),
                            "Option to include the contribution from surface topography on geoid. The default is true.");
          prm.declare_entry("Include CMB topography contribution", "true",
                            Patterns::Bool(),
                            "Option to include the contribution from CMB topography on geoid. The default is true.");
          prm.declare_entry("Maximum degree","20",
                            Patterns::Integer (0),
                            "This parameter can be a random positive integer. However, the value normally should not exceed the maximum "
                            "degree of the initial perturbed temperature field. For example, if the initial temperature uses S40RTS, the "
                            "maximum degree should not be larger than 40.");
          prm.declare_entry("Minimum degree","2",
                            Patterns::Integer (0),
                            "This parameter normally is set to 2 since the perturbed gravitational potential at degree 1 always vanishes "
                            "in a reference frame with the planetary center of mass same as the center of figure.");
          prm.declare_entry("Output data in geographical coordinates", "false",
                            Patterns::Bool(),
                            "Option to output the geoid anomaly in geographical coordinates (latitude and longitude). "
                            "The default is false, so the postprocessor will output the data in geocentric coordinates (x,y,z) as normally.");
          prm.declare_entry("Output text files", "true",
                            Patterns::Bool(),
                            "Whether this postprocessor writes its own geoid text files. "
                            "The spherical-harmonic coefficients are still computed and kept in memory for dependent postprocessors.");
          prm.declare_entry("Reference density for anomaly", "-1e300",
                            Patterns::Double(),
                            "Deprecated.");
          prm.declare_entry("Density anomaly contribution mode", "unspecified",
                            Patterns::Selection("auto|always|never|unspecified"),
                            "Deprecated.");
          prm.declare_entry("Density anomaly tolerance", "-1e300",
                            Patterns::Double(),
                            "Deprecated.");
          prm.declare_entry("Output geoid anomaly coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the geoid anomaly up to the maximum degree. "
                            "The default is false, so the postprocessor will only output the geoid anomaly in grid format. ");
          prm.declare_entry("Output surface topography contribution coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the surface topography contribution "
                            "to the maximum degree. The default is false. ");
          prm.declare_entry("Output CMB topography contribution coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the CMB topography contribution "
                            "to the maximum degree. The default is false. ");
          prm.declare_entry("Output density anomaly contribution coefficients", "false",
                            Patterns::Bool(),
                            "Option to output the spherical harmonic coefficients of the density anomaly contribution to the "
                            "maximum degree. The default is false. ");
          prm.declare_entry("Time between text output", "0.",
                            Patterns::Double(0.),
                            "The simulation time interval between text file outputs.");
          prm.declare_entry("Time steps between text output", "1",
                            Patterns::Integer(0),
                            "The number of time steps between text file outputs.");
          prm.declare_entry("Output gravity anomaly", "false",
                            Patterns::Bool(),
                            "Option to output the free-air gravity anomaly up to the maximum degree. "
                            "The unit of the output is in SI, hence $m/s^2$ ($1mgal = 10^-5 m/s^2$). The default is false. ");

          prm.declare_alias("Output geoid anomaly coefficients","Also output the spherical harmonic coefficients of geoid anomaly");
          prm.declare_alias("Output surface topography contribution coefficients","Also output the spherical harmonic coefficients of surface dynamic topography contribution");
          prm.declare_alias("Output CMB topography contribution coefficients","Also output the spherical harmonic coefficients of CMB dynamic topography contribution");
          prm.declare_alias("Output density anomaly contribution coefficients","Also output the spherical harmonic coefficients of density anomaly contribution");
          prm.declare_alias("Output gravity anomaly","Also output the gravity anomaly");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    Geoid<dim>::parse_parameters (ParameterHandler &prm)
    {
      CitationInfo::add("geoid");
      self_gravity_helper.parse_parameters(prm);
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Geoid");
        {
          include_surface_topo_contribution = prm.get_bool ("Include surface topography contribution");
          include_CMB_topo_contribution = prm.get_bool ("Include CMB topography contribution");
          max_degree = prm.get_integer ("Maximum degree");
          min_degree = prm.get_integer ("Minimum degree");
          output_in_lat_lon = prm.get_bool ("Output data in geographical coordinates");
          output_text_files = prm.get_bool ("Output text files");
          const double legacy_geoid_ref_density = prm.get_double ("Reference density for anomaly");
          std::string mode = "true";
          double tolerance = 0.0;
          double ref_dens = 0.0;

          if (this->get_boundary_traction_manager().template has_matching_active_plugin<BoundaryTraction::PotentialFeedbackTraction<dim>>())
            {
              const auto &pf = this->get_boundary_traction_manager().template get_matching_active_plugin<BoundaryTraction::PotentialFeedbackTraction<dim>>();
              mode = pf.get_settings().include_internal_density_anomalies;
              tolerance = pf.get_settings().internal_density_anomaly_tolerance;
              ref_dens = pf.get_settings().reference_density_for_internal_anomalies;

              const std::string legacy_geoid_mode = prm.get("Density anomaly contribution mode");
              const double legacy_geoid_tolerance = prm.get_double("Density anomaly tolerance");
              if (legacy_geoid_mode != "unspecified" || legacy_geoid_tolerance != -1e300 || legacy_geoid_ref_density != -1e300)
                {
                  this->get_pcout() << "WARNING: Legacy parameters 'Postprocess / Geoid / Reference density for anomaly', "
                                    << "'Density anomaly contribution mode', and 'Density anomaly tolerance' are set, "
                                    << "but they are overridden by the active self-gravity settings under 'Potential feedback / Self gravity'." << std::endl;
                }
            }
          else
            {
              const std::string legacy_geoid_mode = prm.get("Density anomaly contribution mode");
              const double legacy_geoid_tolerance = prm.get_double("Density anomaly tolerance");
              if (legacy_geoid_mode != "unspecified")
                {
                  if (legacy_geoid_mode == "always") mode = "true";
                  else if (legacy_geoid_mode == "never") mode = "false";
                  else if (legacy_geoid_mode == "auto") mode = "auto";
                }
              if (legacy_geoid_tolerance != -1e300)
                {
                  tolerance = legacy_geoid_tolerance;
                }
              if (legacy_geoid_ref_density != -1e300)
                {
                  ref_dens = legacy_geoid_ref_density;
                }
            }

          reference_density = ref_dens;

          if (mode == "true")
            density_anomaly_mode = DensityAnomalyMode::always;
          else if (mode == "false")
            density_anomaly_mode = DensityAnomalyMode::never;
          else if (mode == "auto")
            density_anomaly_mode = DensityAnomalyMode::auto_detect;
          else
            AssertThrow(false, ExcMessage("Unknown density anomaly contribution mode."));

          density_anomaly_tolerance = tolerance;
          output_geoid_anomaly_SH_coes = prm.get_bool ("Output geoid anomaly coefficients");
          output_surface_topo_contribution_SH_coes = prm.get_bool ("Output surface topography contribution coefficients");
          output_CMB_topo_contribution_SH_coes = prm.get_bool ("Output CMB topography contribution coefficients");
          output_density_anomaly_contribution_SH_coes = prm.get_bool ("Output density anomaly contribution coefficients");
          time_between_text_output = prm.get_double("Time between text output");
          if (this->convert_output_to_years())
            time_between_text_output *= constants::year_in_seconds;
          time_steps_between_text_output = prm.get_integer("Time steps between text output");
          output_gravity_anomaly = prm.get_bool ("Output gravity anomaly");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    Geoid<dim>::initialize_simulator (const Simulator<dim> &simulator)
    {
      SimulatorAccess<dim>::initialize_simulator(simulator);
      self_gravity_helper.initialize_simulator(simulator);
    }


    template <int dim>
    void
    Geoid<dim>::initialize ()
    {
      // Find if the included boundaries are active free surfaces
      if (include_surface_topo_contribution == true || include_CMB_topo_contribution == true)
        {
          if (this->get_parameters().mesh_deformation_enabled == true)
            {
              const  types::boundary_id surface_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
              const  types::boundary_id bottom_id = this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

              const std::set<types::boundary_id> mesh_deformation_boundaries = this->get_mesh_deformation_handler().get_active_mesh_deformation_boundary_indicators();

              use_free_surface_topography = mesh_deformation_boundaries.find(surface_id) != mesh_deformation_boundaries.end();
              use_free_CMB_topography = mesh_deformation_boundaries.find(bottom_id) != mesh_deformation_boundaries.end();
            }
        }
      self_gravity_helper.initialize();
    }

  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(Geoid,
                                  "geoid",
                                  "A postprocessor that computes a representation of "
                                  "the geoid based on the density structure in the mantle, "
                                  "as well as the topography at the surface and "
                                  "core mantle boundary (CMB) if desired. The topography is based on the "
                                  "dynamic topography postprocessor in case of no free surface, "
                                  "and based on the real surface from the geometry model in case "
                                  "of a free surface. The geoid is computed "
                                  "from a spherical harmonic expansion, so the geometry "
                                  "of the domain must be a 3d spherical shell.")
  }
}
