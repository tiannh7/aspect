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


#include <aspect/postprocess/surface_love_numbers.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/global.h>
#include <aspect/boundary_traction/potential_feedback_traction.h>
#include <aspect/boundary_traction/spherical_harmonic_load.h>
#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iomanip>
#include <array>
#include <limits>
#include <sstream>


namespace aspect
{
  namespace Postprocess
  {
    namespace
    {
      Tensor<1,3>
      unit_theta_vector(const double theta, const double phi)
      {
        Tensor<1,3> result;
        result[0] = std::cos(theta) * std::cos(phi);
        result[1] = std::cos(theta) * std::sin(phi);
        result[2] = -std::sin(theta);
        return result;
      }



      Tensor<1,3>
      unit_phi_vector(const double phi)
      {
        Tensor<1,3> result;
        result[0] = -std::sin(phi);
        result[1] = std::cos(phi);
        result[2] = 0.0;
        return result;
      }



      std::pair<Tensor<1,3>, Tensor<1,3>>
      spherical_harmonic_surface_gradients(const unsigned int degree,
                                           const unsigned int order,
                                           const double theta,
                                           const double phi)
      {
        const double eps = 1e-6;
        const double theta_minus = std::max(eps, theta - eps);
        const double theta_plus = std::min(numbers::PI - eps, theta + eps);
        const double sin_theta = std::max(std::sin(theta), eps);

        const std::pair<double,double> y_theta_plus =
          Utilities::real_spherical_harmonic(degree, order, theta_plus, phi);
        const std::pair<double,double> y_theta_minus =
          Utilities::real_spherical_harmonic(degree, order, theta_minus, phi);
        const std::pair<double,double> y_phi_plus =
          Utilities::real_spherical_harmonic(degree, order, theta, phi + eps);
        const std::pair<double,double> y_phi_minus =
          Utilities::real_spherical_harmonic(degree, order, theta, phi - eps);

        const double dtheta_cos =
          (y_theta_plus.first - y_theta_minus.first) / (theta_plus - theta_minus);
        const double dtheta_sin =
          (y_theta_plus.second - y_theta_minus.second) / (theta_plus - theta_minus);
        const double dphi_cos =
          (y_phi_plus.first - y_phi_minus.first) / (2.0 * eps);
        const double dphi_sin =
          (y_phi_plus.second - y_phi_minus.second) / (2.0 * eps);

        const Tensor<1,3> e_theta = unit_theta_vector(theta, phi);
        const Tensor<1,3> e_phi = unit_phi_vector(phi);

        return std::make_pair(dtheta_cos * e_theta + dphi_cos / sin_theta * e_phi,
                              dtheta_sin * e_theta + dphi_sin / sin_theta * e_phi);
      }



      unsigned int
      n_spherical_harmonic_coefficients(const unsigned int min_degree,
                                        const unsigned int max_degree)
      {
        unsigned int result = 0;
        for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
          result += degree + 1;
        return result;
      }



      unsigned int
      spherical_harmonic_coefficient_index(const unsigned int min_degree,
                                           const unsigned int degree,
                                           const unsigned int order)
      {
        unsigned int index = 0;
        for (unsigned int l = min_degree; l < degree; ++l)
          index += l + 1;
        return index + order;
      }


      struct DegreeOneProjectionComponent
      {
        std::string name;
        unsigned int order;
        bool sine;
        double external_load = 0.0;
        double surface_deformation = 0.0;
        double cmb_deformation = 0.0;
        double volume = 0.0;
        double total_pre = 0.0;
        double total_post = 0.0;
        double correction_radial = 0.0;
        double correction_poloidal = 0.0;
        double radial_pre = 0.0;
        double radial_post = 0.0;
        double poloidal_pre = 0.0;
        double poloidal_post = 0.0;
        double h_pre = 0.0;
        double h_post = 0.0;
        double l_pre = 0.0;
        double l_post = 0.0;
        double degree_horizontal_l_rms_pre = 0.0;
        double degree_horizontal_l_rms_post = 0.0;
      };


      double
      select_real_coefficient_component(const std::pair<double,double> &coefficient,
                                        const bool sine)
      {
        return sine ? coefficient.second : coefficient.first;
      }



    }



    template <int dim>
    std::pair<std::string,std::string>
    SurfaceLoveNumbers<dim>::execute (TableHandler &statistics)
    {
      if constexpr (dim != 3)
        {
          AssertThrow(false,
                      ExcMessage("The surface love numbers postprocessor is currently only implemented for the 3d spherical shell geometry model."));
          return std::make_pair("", "");
        }
      else
        {
          AssertThrow (Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model()),
                       ExcMessage("The surface love numbers postprocessor is currently only implemented for the 3d spherical shell geometry model."));

          const unsigned int n_coefficients =
            n_spherical_harmonic_coefficients(min_degree, max_degree);
          if (displacement_coecos.size() != n_coefficients)
            {
              displacement_coecos.assign(n_coefficients, 0.0);
              displacement_coesin.assign(n_coefficients, 0.0);
            }
          if (radial_displacement_coecos.size() != n_coefficients)
            {
              radial_displacement_coecos.assign(n_coefficients, 0.0);
              radial_displacement_coesin.assign(n_coefficients, 0.0);
            }

          double timestep = this->get_timestep();
          if (this->get_timestep_number() == 0 && initial_elastic_displacement_time > 0.0)
            timestep = initial_elastic_displacement_time;
          std::vector<double> local_increment_cos(n_coefficients, 0.0);
          std::vector<double> local_increment_sin(n_coefficients, 0.0);
          std::vector<double> local_radial_increment_cos(n_coefficients, 0.0);
          std::vector<double> local_radial_increment_sin(n_coefficients, 0.0);

          if (timestep > 0.0)
            {
              const GeometryModel::SphericalShell<dim> &geometry_model =
                Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>> (this->get_geometry_model());
              const types::boundary_id top_boundary_id =
                geometry_model.translate_symbolic_boundary_name_to_id("top");

              const Quadrature<dim-1> &quadrature_formula =
                this->introspection().face_quadratures.velocities;

              FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                                this->get_fe(),
                                                quadrature_formula,
                                                update_values |
                                                update_quadrature_points |
                                                update_JxW_values);

              std::vector<Tensor<1,dim>> velocity_values(fe_face_values.n_quadrature_points);

              for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  for (const unsigned int face_no : cell->face_indices())
                    if (cell->face(face_no)->at_boundary()
                        &&
                        cell->face(face_no)->boundary_id() == top_boundary_id)
                      {
                        fe_face_values.reinit(cell, face_no);
                        fe_face_values[this->introspection().extractors.velocities].get_function_values(this->get_solution(),
                                                                                                        velocity_values);

                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                          {
                            const Point<dim> position = fe_face_values.quadrature_point(q);
                            const Tensor<1,dim> radial_unit_vector = position / position.norm();
                            const double radial_velocity = velocity_values[q] * radial_unit_vector;
                            const double radial_displacement = timestep * radial_velocity;
                            const Tensor<1,dim> tangential_displacement =
                              timestep * (velocity_values[q] - radial_velocity * radial_unit_vector);

                            Tensor<1,3> tangential_displacement_3d;
                            for (unsigned int d=0; d<dim; ++d)
                              tangential_displacement_3d[d] = tangential_displacement[d];

                            const std::array<double,3> spherical_coordinates =
                              Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                            const double radius = spherical_coordinates[0];
                            const double phi = spherical_coordinates[1];
                            const double theta = spherical_coordinates[2];
                            const double area_weight = fe_face_values.JxW(q) / (radius * radius);

                            unsigned int coefficient_index = 0;
                            for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                              for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                                {
                                  const std::pair<Tensor<1,3>, Tensor<1,3>> gradients =
                                    spherical_harmonic_surface_gradients(degree, order, theta, phi);
                                  const std::pair<double,double> harmonics =
                                    Utilities::real_spherical_harmonic(degree, order, theta, phi);
                                  const double normalization =
                                    (degree > 0 ? static_cast<double>(degree * (degree + 1)) : 1.0);

                                  local_increment_cos[coefficient_index] +=
                                    (tangential_displacement_3d * gradients.first) * area_weight / normalization;
                                  local_increment_sin[coefficient_index] +=
                                    (tangential_displacement_3d * gradients.second) * area_weight / normalization;
                                  local_radial_increment_cos[coefficient_index] +=
                                    radial_displacement * harmonics.first * area_weight;
                                  local_radial_increment_sin[coefficient_index] +=
                                    radial_displacement * harmonics.second * area_weight;
                                }
                          }
                      }
            }

          std::vector<double> global_increment_cos(n_coefficients, 0.0);
          std::vector<double> global_increment_sin(n_coefficients, 0.0);
          std::vector<double> global_radial_increment_cos(n_coefficients, 0.0);
          std::vector<double> global_radial_increment_sin(n_coefficients, 0.0);
          Utilities::MPI::sum(local_increment_cos,
                              this->get_mpi_communicator(),
                              global_increment_cos);
          Utilities::MPI::sum(local_increment_sin,
                              this->get_mpi_communicator(),
                              global_increment_sin);
          Utilities::MPI::sum(local_radial_increment_cos,
                              this->get_mpi_communicator(),
                              global_radial_increment_cos);
          Utilities::MPI::sum(local_radial_increment_sin,
                              this->get_mpi_communicator(),
                              global_radial_increment_sin);

          for (unsigned int i=0; i<n_coefficients; ++i)
            {
              displacement_coecos[i] += global_increment_cos[i];
              displacement_coesin[i] += global_increment_sin[i];
              radial_displacement_coecos[i] += global_radial_increment_cos[i];
              radial_displacement_coesin[i] += global_radial_increment_sin[i];
            }

          if (degree_one_displacement_reference_frame == "nullspace-restored"
              && min_degree <= 1
              && max_degree >= 1
              && timestep > 0.0)
            {
              const Tensor<1,dim> removed_translation_velocity =
                this->get_last_removed_net_translation();
              const Tensor<1,dim> removed_translation_displacement =
                timestep * removed_translation_velocity;
              const double normalization =
                std::sqrt(3.0 / (4.0 * numbers::PI));

              const unsigned int degree_one_order_zero_index =
                spherical_harmonic_coefficient_index(min_degree, 1, 0);
              const unsigned int degree_one_order_one_index =
                spherical_harmonic_coefficient_index(min_degree, 1, 1);

              displacement_coecos[degree_one_order_zero_index] +=
                removed_translation_displacement[2] / normalization;
              displacement_coecos[degree_one_order_one_index] +=
                -removed_translation_displacement[0] / normalization;
              displacement_coesin[degree_one_order_one_index] +=
                -removed_translation_displacement[1] / normalization;
              radial_displacement_coecos[degree_one_order_zero_index] +=
                removed_translation_displacement[2] / normalization;
              radial_displacement_coecos[degree_one_order_one_index] +=
                -removed_translation_displacement[0] / normalization;
              radial_displacement_coesin[degree_one_order_one_index] +=
                -removed_translation_displacement[1] / normalization;
            }

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

          if (output_needed)
            {
              Utilities::create_directory(this->get_output_directory() + "surface_love_numbers/",
                                          this->get_mpi_communicator(),
                                          true);

              double top_total_normal_traction_10 = 0.0;
              double bottom_total_normal_traction_10 = 0.0;
              double top_external_load_normal_traction_10 = 0.0;
              double bottom_external_load_normal_traction_10 = 0.0;
              double top_self_gravity_plus_compensation_normal_traction_10 = 0.0;
              double bottom_self_gravity_plus_compensation_normal_traction_10 = 0.0;
              double top_equivalent_height_from_total_traction_10 = 0.0;
              double bottom_equivalent_height_from_total_traction_10 = 0.0;
              double top_y10_normalization_integral = 0.0;
              double bottom_y10_normalization_integral = 0.0;
              double top_gravity = 0.0;
              double bottom_gravity = 0.0;
              double surface_density_jump = 0.0;
              double cmb_density_jump = 0.0;
              double surface_final_rhs_10 = 0.0;
              double cmb_intermediate_compensation_rhs_10 = 0.0;
              double cmb_potential_append_rhs_10 = 0.0;
              double cmb_final_rhs_10 = 0.0;
              double raw_surface_u10_from_stokes_displacement = 0.0;
              double raw_cmb_u10_from_stokes_displacement = 0.0;
              double surface_deformation_height_10_used_by_self_gravity = 0.0;
              double cmb_deformation_height_10_used_by_self_gravity = 0.0;
              Tensor<1,dim> removed_net_translation_velocity;
              Tensor<1,dim> removed_net_translation_displacement;
              double removed_net_translation_u10_equivalent = 0.0;

              if (degree_one_boundary_traction_rhs_diagnostic
                  && min_degree <= 1
                  && max_degree >= 1)
                {
                  removed_net_translation_velocity =
                    this->get_last_removed_net_translation();
                  removed_net_translation_displacement =
                    timestep * removed_net_translation_velocity;
                  removed_net_translation_u10_equivalent =
                    removed_net_translation_displacement[2]
                    / std::sqrt(3.0 / (4.0 * numbers::PI));

                  const GeometryModel::SphericalShell<dim> &geometry_model =
                    Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>> (this->get_geometry_model());
                  const types::boundary_id top_boundary_id =
                    geometry_model.translate_symbolic_boundary_name_to_id("top");
                  const types::boundary_id bottom_boundary_id =
                    geometry_model.translate_symbolic_boundary_name_to_id("bottom");

                  Point<dim> top_reference_point;
                  Point<dim> bottom_reference_point;
                  top_reference_point[dim-1] = geometry_model.outer_radius();
                  bottom_reference_point[dim-1] = geometry_model.inner_radius();
                  top_gravity =
                    this->get_gravity_model().gravity_vector(top_reference_point).norm();
                  bottom_gravity =
                    this->get_gravity_model().gravity_vector(bottom_reference_point).norm();

                  const auto &traction_manager =
                    this->get_boundary_traction_manager();
                  const PotentialFeedback::SelfGravitation<dim> *active_self_gravity =
                    &self_gravity_helper;
                  if (traction_manager.template has_matching_active_plugin<
                      PotentialFeedback::SelfGravitation<dim>>())
                    active_self_gravity =
                      &traction_manager.template get_matching_active_plugin<
                      PotentialFeedback::SelfGravitation<dim>>();
                  else if (traction_manager.template has_matching_active_plugin<
                           BoundaryTraction::PotentialFeedbackTraction<dim>>())
                    {
                      const auto &potential_feedback =
                        traction_manager.template get_matching_active_plugin<
                        BoundaryTraction::PotentialFeedbackTraction<dim>>();
                      if (potential_feedback.has_self_gravity_feedback())
                        active_self_gravity =
                          &potential_feedback.get_self_gravity();
                    }
                  surface_density_jump =
                    active_self_gravity->surface_density_jump();
                  cmb_density_jump =
                    active_self_gravity->cmb_density_jump();
                  if (active_self_gravity
                      ->has_citcomsve_degree_one_load_replay_diagnostic())
                    {
                      cmb_intermediate_compensation_rhs_10 =
                        active_self_gravity
                        ->citcomsve_degree_one_cmb_intermediate_compensation_rhs_10();
                      cmb_potential_append_rhs_10 =
                        active_self_gravity
                        ->citcomsve_degree_one_cmb_potential_append_rhs_10();
                    }

                  const auto &traction_boundary_ids =
                    traction_manager.get_prescribed_boundary_traction_indicators();
                  const bool has_external_load =
                    traction_manager.template has_matching_active_plugin<
                    BoundaryTraction::SphericalHarmonicLoad<dim>>();
                  const BoundaryTraction::SphericalHarmonicLoad<dim> *external_load = nullptr;
                  if (has_external_load)
                    external_load = &traction_manager.template get_matching_active_plugin<
                                    BoundaryTraction::SphericalHarmonicLoad<dim>>();

                  const Quadrature<dim-1> &quadrature_formula =
                    this->introspection().face_quadratures.velocities;

                  FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                                    this->get_fe(),
                                                    quadrature_formula,
                                                    update_values |
                                                    update_quadrature_points |
                                                    update_normal_vectors |
                                                    update_JxW_values);
                  std::vector<Tensor<1,dim>> velocity_values(
                    fe_face_values.n_quadrature_points);

                  enum
                  {
                    top_total = 0,
                    bottom_total = 1,
                    top_external = 2,
                    bottom_external = 3,
                    top_norm = 4,
                    bottom_norm = 5,
                    top_raw_u = 6,
                    bottom_raw_u = 7,
                    n_values = 8
                  };
                  double values[n_values] = {};

                  for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                    if (cell->is_locally_owned())
                      for (const unsigned int face_no : cell->face_indices())
                        if (cell->face(face_no)->at_boundary())
                          {
                            const types::boundary_id boundary_id =
                              cell->face(face_no)->boundary_id();
                            const bool is_top = boundary_id == top_boundary_id;
                            const bool is_bottom = boundary_id == bottom_boundary_id;
                            if (!is_top && !is_bottom)
                              continue;
                            if (traction_boundary_ids.find(boundary_id) ==
                                traction_boundary_ids.end())
                              continue;

                            fe_face_values.reinit(cell, face_no);
                            fe_face_values[this->introspection().extractors.velocities]
                            .get_function_values(this->get_solution(),
                                                 velocity_values);

                            for (unsigned int q=0;
                                 q<fe_face_values.n_quadrature_points;
                                 ++q)
                              {
                                const Point<dim> position =
                                  fe_face_values.quadrature_point(q);
                                const Tensor<1,dim> normal =
                                  fe_face_values.normal_vector(q);
                                const std::array<double,dim> spherical_coordinates =
                                  Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                                const double radius = spherical_coordinates[0];
                                const double phi = spherical_coordinates[1];
                                const double theta = spherical_coordinates[2];
                                const double y10 =
                                  Utilities::real_spherical_harmonic(1, 0,
                                                                     theta,
                                                                     phi).first;
                                const double area_weight =
                                  fe_face_values.JxW(q) / (radius * radius);
                                const Tensor<1,dim> radial_unit =
                                  position / radius;
                                const double radial_displacement =
                                  timestep * (velocity_values[q] * radial_unit);

                                const Tensor<1,dim> total_traction =
                                  traction_manager.boundary_traction(boundary_id,
                                                                     position,
                                                                     normal);
                                const double total_normal_traction =
                                  total_traction * normal;

                                Tensor<1,dim> external_traction;
                                if (external_load != nullptr && is_top)
                                  external_traction =
                                    external_load->boundary_traction(boundary_id,
                                                                     position,
                                                                     normal);
                                const double external_normal_traction =
                                  external_traction * normal;

                                const unsigned int value_index =
                                  (is_top ? top_total : bottom_total);
                                const unsigned int external_index =
                                  (is_top ? top_external : bottom_external);
                                const unsigned int norm_index =
                                  (is_top ? top_norm : bottom_norm);

                                values[value_index] +=
                                  total_normal_traction * y10 * area_weight;
                                values[external_index] +=
                                  external_normal_traction * y10 * area_weight;
                                values[norm_index] += y10 * y10 * area_weight;
                                values[(is_top ? top_raw_u : bottom_raw_u)] +=
                                  radial_displacement * y10 * area_weight;
                              }
                          }

                  Utilities::MPI::sum<double, n_values>(values,
                                                        this->get_mpi_communicator(),
                                                        values);

                  if (values[top_norm] > 0.0)
                    {
                      top_y10_normalization_integral = values[top_norm];
                      top_total_normal_traction_10 =
                        values[top_total] / values[top_norm];
                      top_external_load_normal_traction_10 =
                        values[top_external] / values[top_norm];
                      top_self_gravity_plus_compensation_normal_traction_10 =
                        top_total_normal_traction_10
                        - top_external_load_normal_traction_10;
                      raw_surface_u10_from_stokes_displacement =
                        values[top_raw_u] / values[top_norm];
                      surface_deformation_height_10_used_by_self_gravity =
                        raw_surface_u10_from_stokes_displacement;
                    }
                  if (values[bottom_norm] > 0.0)
                    {
                      bottom_y10_normalization_integral = values[bottom_norm];
                      bottom_total_normal_traction_10 =
                        values[bottom_total] / values[bottom_norm];
                      bottom_external_load_normal_traction_10 =
                        values[bottom_external] / values[bottom_norm];
                      bottom_self_gravity_plus_compensation_normal_traction_10 =
                        bottom_total_normal_traction_10
                        - bottom_external_load_normal_traction_10;
                      raw_cmb_u10_from_stokes_displacement =
                        values[bottom_raw_u] / values[bottom_norm];
                      cmb_deformation_height_10_used_by_self_gravity =
                        raw_cmb_u10_from_stokes_displacement;
                    }

                  if (surface_density_jump * top_gravity != 0.0)
                    top_equivalent_height_from_total_traction_10 =
                      top_total_normal_traction_10
                      / (surface_density_jump * top_gravity);
                  if (cmb_density_jump * bottom_gravity != 0.0)
                    bottom_equivalent_height_from_total_traction_10 =
                      bottom_total_normal_traction_10
                      / (cmb_density_jump * bottom_gravity);

                  surface_final_rhs_10 =
                    top_equivalent_height_from_total_traction_10;
                  cmb_final_rhs_10 =
                    -bottom_equivalent_height_from_total_traction_10;
                }

              const PotentialFeedback::SelfGravitation<dim> *self_gravity = nullptr;
              const BoundaryTraction::PotentialFeedbackTraction<dim> *potential_feedback = nullptr;
              bool use_boundary_potential = false;
              unsigned int self_gravity_coefficient_min_degree = 0;
              std::pair<std::vector<double>,std::vector<double>> SH_density_coes;
              std::pair<double, std::pair<std::vector<double>,std::vector<double>>> SH_surface_topo_coes;
              std::pair<double, std::pair<std::vector<double>,std::vector<double>>> SH_CMB_topo_coes;
              double surface_radius = 0.0;
              double surface_gravity = 0.0;
              double inner_radius = 0.0;
              double CMB_delta_rho = 0.0;

              if (output_coefficients)
                {
                  const Point<dim> surface_point =
                    this->get_geometry_model().representative_point(1.0);
                  surface_radius = surface_point.norm();
                  surface_gravity =
                    this->get_gravity_model().gravity_vector(surface_point).norm();

                  const auto &traction_manager = this->get_boundary_traction_manager();
                  const bool use_self_gravity =
                    traction_manager.template has_matching_active_plugin<
                    PotentialFeedback::SelfGravitation<dim>>();
                  const bool use_potential_feedback =
                    traction_manager.template has_matching_active_plugin<
                    BoundaryTraction::PotentialFeedbackTraction<dim>>();

                  if (use_self_gravity)
                    self_gravity = &traction_manager.template get_matching_active_plugin<
                                   PotentialFeedback::SelfGravitation<dim>>();
                  else if (use_potential_feedback)
                    {
                      const auto &pf = traction_manager.template get_matching_active_plugin<
                                       BoundaryTraction::PotentialFeedbackTraction<dim>>();
                      if (pf.has_self_gravity_feedback())
                        {
                          potential_feedback = &pf;
                          self_gravity = &pf.get_self_gravity();
                        }
                    }

                  if (self_gravity == nullptr)
                    self_gravity = &self_gravity_helper;

                  use_boundary_potential =
                    use_self_gravity || potential_feedback != nullptr;
                  self_gravity_coefficient_min_degree =
                    self_gravity->minimum_degree();
                  SH_density_coes =
                    self_gravity->compute_internal_density_potential(surface_radius);

                  const GeometryModel::SphericalShell<dim> &geometry_model =
                    Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>> (this->get_geometry_model());
                  inner_radius = geometry_model.inner_radius();
                  CMB_delta_rho = self_gravity->cmb_density_jump();

                  if (!use_boundary_potential)
                    {
                      const auto SH_topo_coes =
                        self_gravity->compute_topography_potential(surface_radius, inner_radius);
                      SH_surface_topo_coes = SH_topo_coes.first;
                      SH_CMB_topo_coes = SH_topo_coes.second;
                    }
                }

              if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
                {
                  const std::string timestep_suffix =
                    "." + Utilities::int_to_string(this->get_timestep_number(), 5);
                  const std::string output_directory =
                    this->get_output_directory() + "surface_love_numbers/";

                  if (degree_one_boundary_traction_rhs_diagnostic
                      && min_degree <= 1
                      && max_degree >= 1)
                    {
                      std::ofstream rhs_output(
                        output_directory
                        + "degree1_boundary_traction_rhs_diagnostic"
                        + timestep_suffix);
                      rhs_output
                          << "# ASPECT l1m0 boundary traction RHS diagnostic\n";
                      rhs_output
                          << "# total normal traction is traction dot outward_normal, projected onto normalized Y10.\n";
                      rhs_output
                          << "# total columns are from boundary_traction_manager.boundary_traction(), the same callback used by StokesBoundaryTraction.\n";
                      rhs_output
                          << "# external_load columns call the active Spherical harmonic load plugin on the top boundary; self_gravity_plus_compensation is total minus external_load.\n";
                      rhs_output
                          << "# raw U10 columns are Stokes velocity times the initial elastic displacement time, before any degree-1 output-frame correction.\n";
                      rhs_output
                          << "# raw_top/cmb_radial_displacement_Y10_outward_positive are aliases for the raw U10 columns with explicit sign convention.\n";
                      rhs_output
                          << "# removed_net_translation_* is the velocity/displacement subtracted by ASPECT's net-translation nullspace removal.\n";
                      rhs_output
                          << "# deformation_height_used_by_self_gravity columns are the same raw radial coefficients for these diagnostic no-feedback compliance runs; when self-gravity feedback is active, the self_gravity diagnostic gives the projected mesh-displacement coefficients used by the feedback operator.\n";
                      rhs_output
                          << "# timestep: " << this->get_timestep_number() << "\n";
                      rhs_output
                          << "# time: " << std::setprecision(16)
                          << this->get_time() << "\n";
                      rhs_output
                          << "top_total_normal_traction_10 "
                          << "bottom_total_normal_traction_10 "
                          << "top_equivalent_height_from_total_traction_10 "
                          << "bottom_equivalent_height_from_total_traction_10 "
                          << "top_external_load_normal_traction_10 "
                          << "bottom_external_load_normal_traction_10 "
                          << "top_self_gravity_plus_compensation_normal_traction_10 "
                          << "bottom_self_gravity_plus_compensation_normal_traction_10 "
                          << "top_y10_normalization_integral "
                          << "bottom_y10_normalization_integral "
                          << "surface_density_jump "
                          << "cmb_density_jump "
                          << "top_gravity "
                          << "bottom_gravity "
                          << "surface_final_rhs_10 "
                          << "cmb_intermediate_compensation_rhs_10 "
                          << "cmb_potential_append_rhs_10 "
                          << "cmb_final_rhs_10 "
                          << "raw_surface_U10_before_CM_or_output_shift "
                          << "raw_cmb_U10_before_CM_or_output_shift "
                          << "raw_top_radial_displacement_Y10_outward_positive "
                          << "raw_cmb_radial_displacement_Y10_outward_positive "
                          << "removed_net_translation_velocity_x "
                          << "removed_net_translation_velocity_y "
                          << "removed_net_translation_velocity_z "
                          << "removed_net_translation_displacement_x "
                          << "removed_net_translation_displacement_y "
                          << "removed_net_translation_displacement_z "
                          << "removed_net_translation_U10_equivalent "
                          << "surface_deformation_height_10_used_by_self_gravity "
                          << "cmb_deformation_height_10_used_by_self_gravity\n";
                      rhs_output
                          << std::setprecision(16)
                          << top_total_normal_traction_10 << ' '
                          << bottom_total_normal_traction_10 << ' '
                          << top_equivalent_height_from_total_traction_10 << ' '
                          << bottom_equivalent_height_from_total_traction_10 << ' '
                          << top_external_load_normal_traction_10 << ' '
                          << bottom_external_load_normal_traction_10 << ' '
                          << top_self_gravity_plus_compensation_normal_traction_10 << ' '
                          << bottom_self_gravity_plus_compensation_normal_traction_10 << ' '
                          << top_y10_normalization_integral << ' '
                          << bottom_y10_normalization_integral << ' '
                          << surface_density_jump << ' '
                          << cmb_density_jump << ' '
                          << top_gravity << ' '
                          << bottom_gravity << ' '
                          << surface_final_rhs_10 << ' '
                          << cmb_intermediate_compensation_rhs_10 << ' '
                          << cmb_potential_append_rhs_10 << ' '
                          << cmb_final_rhs_10 << ' '
                          << raw_surface_u10_from_stokes_displacement << ' '
                          << raw_cmb_u10_from_stokes_displacement << ' '
                          << raw_surface_u10_from_stokes_displacement << ' '
                          << raw_cmb_u10_from_stokes_displacement << ' '
                          << removed_net_translation_velocity[0] << ' '
                          << removed_net_translation_velocity[1] << ' '
                          << removed_net_translation_velocity[2] << ' '
                          << removed_net_translation_displacement[0] << ' '
                          << removed_net_translation_displacement[1] << ' '
                          << removed_net_translation_displacement[2] << ' '
                          << removed_net_translation_u10_equivalent << ' '
                          << surface_deformation_height_10_used_by_self_gravity << ' '
                          << cmb_deformation_height_10_used_by_self_gravity
                          << '\n';
                    }

                  std::ofstream displacement_output(output_directory +
                                                    "surface_tangential_displacement_SH_coefficients" +
                                                    timestep_suffix);
                  displacement_output << "# degree order cosine_coefficient sine_coefficient\n";
                  displacement_output << "# field: cumulative poloidal tangential displacement coefficient V_lm, m\n";
                  displacement_output << "# projection: integral(u_t dot grad_s Y_lm) / l(l+1)\n";
                  displacement_output << "# degree_1_displacement_reference_frame: "
                                      << degree_one_displacement_reference_frame
                                      << "\n";
                  displacement_output << "# time: " << std::setprecision(16) << this->get_time() << "\n";
                  displacement_output << "# timestep: " << this->get_timestep_number() << "\n";

                  unsigned int coefficient_index = 0;
                  for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                    for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                      {
                        displacement_output << degree << ' '
                                            << order << ' '
                                            << std::setprecision(16) << displacement_coecos[coefficient_index] << ' '
                                            << std::setprecision(16) << displacement_coesin[coefficient_index] << '\n';
                      }

                  std::ofstream radial_displacement_output(output_directory +
                                                           "surface_radial_displacement_SH_coefficients" +
                                                           timestep_suffix);
                  radial_displacement_output << "# degree order cosine_coefficient sine_coefficient\n";
                  radial_displacement_output << "# field: cumulative radial displacement coefficient U_lm, m\n";
                  radial_displacement_output << "# projection: integral(u_r Y_lm)\n";
                  radial_displacement_output << "# degree_1_displacement_reference_frame: "
                                             << degree_one_displacement_reference_frame
                                             << "\n";
                  radial_displacement_output << "# time: " << std::setprecision(16) << this->get_time() << "\n";
                  radial_displacement_output << "# timestep: " << this->get_timestep_number() << "\n";

                  coefficient_index = 0;
                  for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                    for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                      {
                        radial_displacement_output << degree << ' '
                                                   << order << ' '
                                                   << std::setprecision(16) << radial_displacement_coecos[coefficient_index] << ' '
                                                   << std::setprecision(16) << radial_displacement_coesin[coefficient_index] << '\n';
                      }

                  if (output_coefficients)
                    {
                      const auto self_gravity_vector_coefficient =
                        [self_gravity_coefficient_min_degree]
                        (const std::pair<std::vector<double>, std::vector<double>> &coefficients,
                         const unsigned int degree,
                         const unsigned int order,
                         const bool sine)
                      {
                        if (degree < self_gravity_coefficient_min_degree
                            || order > degree)
                          return 0.0;

                        const unsigned int index =
                          spherical_harmonic_coefficient_index(
                            self_gravity_coefficient_min_degree,
                            degree,
                            order);
                        const std::vector<double> &values =
                          (sine ? coefficients.second : coefficients.first);
                        if (index >= values.size())
                          return 0.0;

                        return values[index];
                      };

                      const double surface_delta_rho = self_gravity->surface_density_jump();
                      const auto surface_mass_potential_coefficient =
                        [self_gravity, potential_feedback, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential)
                          return std::pair<double,double> {0.0, 0.0};
                        if (potential_feedback != nullptr)
                          return potential_feedback->surface_mass_potential_coefficient(degree, order);
                        return self_gravity->surface_mass_potential_coefficient(degree, order);
                      };
                      const auto external_load_surface_potential_coefficient =
                        [self_gravity, potential_feedback, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential)
                          return std::pair<double,double> {0.0, 0.0};
                        if (potential_feedback != nullptr)
                          return potential_feedback->external_load_surface_potential_coefficient(degree, order);
                        return self_gravity->external_load_surface_potential_coefficient(degree, order);
                      };
                      const auto surface_deformation_mass_potential_coefficient =
                        [self_gravity, potential_feedback, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential)
                          return std::pair<double,double> {0.0, 0.0};
                        if (potential_feedback != nullptr)
                          return potential_feedback->surface_deformation_mass_potential_coefficient(degree, order);
                        return self_gravity->surface_deformation_mass_potential_coefficient(degree, order);
                      };
                      const auto cmb_mass_potential_coefficient =
                        [self_gravity, potential_feedback, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential)
                          return std::pair<double,double> {0.0, 0.0};
                        if (potential_feedback != nullptr)
                          return potential_feedback->cmb_mass_potential_coefficient(degree, order);
                        return self_gravity->cmb_mass_potential_coefficient(degree, order);
                      };
                      const auto tidal_surface_potential_coefficient =
                        [self_gravity, potential_feedback, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential)
                          return std::pair<double,double> {0.0, 0.0};
                        if (potential_feedback != nullptr)
                          return potential_feedback->tidal_surface_potential_coefficient(degree, order);
                        return self_gravity->tidal_surface_potential_coefficient(degree, order);
                      };
                      const auto rotational_surface_potential_coefficient =
                        [potential_feedback, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential || potential_feedback == nullptr)
                          return std::pair<double,double> {0.0, 0.0};
                        return potential_feedback->rotational_surface_potential_coefficient(
                                 degree, order);
                      };
                      const auto reference_frame_surface_potential_coefficient =
                        [self_gravity, use_boundary_potential]
                        (const unsigned int degree,
                         const unsigned int order)
                      {
                        if (!use_boundary_potential)
                          return std::pair<double,double> {0.0, 0.0};
                        return self_gravity->reference_frame_surface_potential_coefficient(degree, order);
                      };
                      const auto adjusted_tangential_displacement =
                        [this, self_gravity]
                        (const unsigned int degree,
                         const unsigned int order,
                         const unsigned int coefficient_index)
                      {
                        double tangential_displacement_cos =
                          displacement_coecos[coefficient_index];
                        double tangential_displacement_sin =
                          displacement_coesin[coefficient_index];

                        if (degree_one_displacement_reference_frame == "citcomsve-deformation-cm"
                            && degree == 1
                            && self_gravity != nullptr)
                          {
                            const Tensor<1,dim> deformation_cm =
                              self_gravity->get_deformation_cm_displacement_increment();
                            const double normalization =
                              std::sqrt(3.0 / (4.0 * numbers::PI));

                            if (order == 0)
                              tangential_displacement_cos -=
                                degree_one_cm_displacement_scale
                                * deformation_cm[2] / normalization;
                            else if (order == 1)
                              {
                                tangential_displacement_cos +=
                                  degree_one_cm_displacement_scale
                                  * deformation_cm[0] / normalization;
                                tangential_displacement_sin +=
                                  degree_one_cm_displacement_scale
                                  * deformation_cm[1] / normalization;
                              }
                          }

                        return std::make_pair(tangential_displacement_cos,
                                              tangential_displacement_sin);
                      };

                      const auto adjusted_radial_displacement =
                        [this, self_gravity]
                        (const unsigned int degree,
                         const unsigned int order,
                         const unsigned int coefficient_index)
                      {
                        double radial_displacement_cos =
                          radial_displacement_coecos[coefficient_index];
                        double radial_displacement_sin =
                          radial_displacement_coesin[coefficient_index];

                        if (degree_one_displacement_reference_frame == "citcomsve-deformation-cm"
                            && degree == 1
                            && self_gravity != nullptr)
                          {
                            const Tensor<1,dim> deformation_cm =
                              self_gravity->get_deformation_cm_displacement_increment();
                            const double normalization =
                              std::sqrt(3.0 / (4.0 * numbers::PI));

                            if (order == 0)
                              radial_displacement_cos -=
                                degree_one_cm_displacement_scale
                                * deformation_cm[2] / normalization;
                            else if (order == 1)
                              {
                                radial_displacement_cos +=
                                  degree_one_cm_displacement_scale
                                  * deformation_cm[0] / normalization;
                                radial_displacement_sin +=
                                  degree_one_cm_displacement_scale
                                  * deformation_cm[1] / normalization;
                              }
                          }

                        return std::make_pair(radial_displacement_cos,
                                              radial_displacement_sin);
                      };

                      const bool degree_one_projection_enabled =
                        degree_one_mass_moment_projection_prototype
                        && load_degree == 1
                        && load_order == 0
                        && min_degree <= 1
                        && max_degree >= 1
                        && use_boundary_potential;
                      double degree_one_projection_phi_sensitivity = 0.0;
                      double degree_one_projection_radial_correction = 0.0;
                      double degree_one_projection_poloidal_correction = 0.0;
                      double degree_one_projection_phi_total_pre = 0.0;

                      if (degree_one_projection_enabled)
                        {
                          const std::pair<double,double> external_load =
                            external_load_surface_potential_coefficient(1, 0);
                          const std::pair<double,double> surface_deformation =
                            surface_deformation_mass_potential_coefficient(1, 0);
                          const std::pair<double,double> cmb_deformation =
                            cmb_mass_potential_coefficient(1, 0);
                          const double volume =
                            (4.0 * numbers::PI * constants::big_g
                             / (surface_gravity * 3.0))
                            * self_gravity_vector_coefficient(
                              SH_density_coes,
                              1,
                              0,
                              false);

                          degree_one_projection_phi_total_pre =
                            external_load.first
                            + surface_deformation.first
                            + cmb_deformation.first
                            + volume;
                          degree_one_projection_phi_sensitivity =
                            4.0 * numbers::PI * constants::big_g
                            * surface_delta_rho * surface_radius
                            / (surface_gravity * 3.0);

                          AssertThrow(std::abs(degree_one_projection_phi_sensitivity)
                                      > std::numeric_limits<double>::epsilon(),
                                      ExcMessage("The degree-1 mass-moment "
                                                 "projection prototype needs a "
                                                 "nonzero surface density jump."));
                          degree_one_projection_radial_correction =
                            -degree_one_projection_phi_total_pre
                            / degree_one_projection_phi_sensitivity;
                          degree_one_projection_poloidal_correction =
                            degree_one_mass_moment_projection_poloidal_ratio
                            * degree_one_projection_radial_correction;
                        }

                      const auto projected_tangential_displacement =
                        [adjusted_tangential_displacement,
                         degree_one_projection_enabled,
                         degree_one_projection_poloidal_correction]
                        (const unsigned int degree,
                         const unsigned int order,
                         const unsigned int coefficient_index)
                      {
                        std::pair<double,double> tangential_displacement =
                          adjusted_tangential_displacement(degree,
                                                           order,
                                                           coefficient_index);
                        if (degree_one_projection_enabled
                            && degree == 1
                            && order == 0)
                          tangential_displacement.first +=
                            degree_one_projection_poloidal_correction;

                        return tangential_displacement;
                      };

                      const auto projected_radial_displacement =
                        [adjusted_radial_displacement,
                         degree_one_projection_enabled,
                         degree_one_projection_radial_correction]
                        (const unsigned int degree,
                         const unsigned int order,
                         const unsigned int coefficient_index)
                      {
                        std::pair<double,double> radial_displacement =
                          adjusted_radial_displacement(degree,
                                                       order,
                                                       coefficient_index);
                        if (degree_one_projection_enabled
                            && degree == 1
                            && order == 0)
                          radial_displacement.first +=
                            degree_one_projection_radial_correction;

                        return radial_displacement;
                      };

                      std::vector<double> horizontal_love_rms_by_degree(max_degree + 1, 0.0);
                      std::vector<double> projected_horizontal_love_rms_by_degree(max_degree + 1, 0.0);
                      coefficient_index = 0;
                      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                        {
                          const double displacement_to_love_scale =
                            (2.0 * degree + 1.0) * surface_gravity
                            / (4.0 * numbers::PI * constants::big_g
                               * load_density * surface_radius);
                          double degree_love_norm_square = 0.0;
                          double projected_degree_love_norm_square = 0.0;
                          for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                            {
                              const std::pair<double,double> tangential_displacement =
                                adjusted_tangential_displacement(degree,
                                                                 order,
                                                                 coefficient_index);
                              const std::pair<double,double> projected_tangential =
                                projected_tangential_displacement(degree,
                                                                  order,
                                                                  coefficient_index);
                              const double l_cos =
                                tangential_displacement.first / load_height
                                * displacement_to_love_scale;
                              const double l_sin =
                                tangential_displacement.second / load_height
                                * displacement_to_love_scale;
                              const double projected_l_cos =
                                projected_tangential.first / load_height
                                * displacement_to_love_scale;
                              const double projected_l_sin =
                                projected_tangential.second / load_height
                                * displacement_to_love_scale;
                              degree_love_norm_square += l_cos * l_cos + l_sin * l_sin;
                              projected_degree_love_norm_square +=
                                projected_l_cos * projected_l_cos
                                + projected_l_sin * projected_l_sin;
                            }
                          horizontal_love_rms_by_degree[degree] =
                            std::sqrt(degree_love_norm_square);
                          projected_horizontal_love_rms_by_degree[degree] =
                            std::sqrt(projected_degree_love_norm_square);
                        }

                      std::vector<DegreeOneProjectionComponent> projection_components;
                      if (degree_one_projection_enabled)
                        {
                          const std::vector<DegreeOneProjectionComponent> component_templates =
                          {
                            {"Phi_10_total", 0, false},
                            {"Phi_11c_total", 1, false},
                            {"Phi_11s_total", 1, true}
                          };

                          const double displacement_to_love_scale =
                            3.0 * surface_gravity
                            / (4.0 * numbers::PI * constants::big_g
                               * load_density * surface_radius);
                          for (const DegreeOneProjectionComponent &component_template :
                               component_templates)
                            {
                              DegreeOneProjectionComponent component =
                                component_template;
                              const unsigned int index =
                                spherical_harmonic_coefficient_index(
                                  min_degree, 1, component.order);
                              component.external_load =
                                select_real_coefficient_component(
                                  external_load_surface_potential_coefficient(
                                    1, component.order),
                                  component.sine);
                              component.surface_deformation =
                                select_real_coefficient_component(
                                  surface_deformation_mass_potential_coefficient(
                                    1, component.order),
                                  component.sine);
                              component.cmb_deformation =
                                select_real_coefficient_component(
                                  cmb_mass_potential_coefficient(
                                    1, component.order),
                                  component.sine);
                              component.volume =
                                (4.0 * numbers::PI * constants::big_g
                                 / (surface_gravity * 3.0))
                                * self_gravity_vector_coefficient(
                                  SH_density_coes,
                                  1,
                                  component.order,
                                  component.sine);
                              component.total_pre =
                                component.external_load
                                + component.surface_deformation
                                + component.cmb_deformation
                                + component.volume;
                              component.correction_radial =
                                (!component.sine && component.order == 0
                                 ? degree_one_projection_radial_correction
                                 : 0.0);
                              component.correction_poloidal =
                                (!component.sine && component.order == 0
                                 ? degree_one_projection_poloidal_correction
                                 : 0.0);
                              component.total_post =
                                component.total_pre
                                + degree_one_projection_phi_sensitivity
                                * component.correction_radial;

                              const std::pair<double,double> radial_pre =
                                adjusted_radial_displacement(1,
                                                             component.order,
                                                             index);
                              const std::pair<double,double> radial_post =
                                projected_radial_displacement(1,
                                                              component.order,
                                                              index);
                              const std::pair<double,double> poloidal_pre =
                                adjusted_tangential_displacement(1,
                                                                 component.order,
                                                                 index);
                              const std::pair<double,double> poloidal_post =
                                projected_tangential_displacement(1,
                                                                  component.order,
                                                                  index);
                              component.radial_pre =
                                select_real_coefficient_component(radial_pre,
                                                                  component.sine);
                              component.radial_post =
                                select_real_coefficient_component(radial_post,
                                                                  component.sine);
                              component.poloidal_pre =
                                select_real_coefficient_component(poloidal_pre,
                                                                  component.sine);
                              component.poloidal_post =
                                select_real_coefficient_component(poloidal_post,
                                                                  component.sine);
                              component.h_pre =
                                component.radial_pre / load_height
                                * displacement_to_love_scale;
                              component.h_post =
                                component.radial_post / load_height
                                * displacement_to_love_scale;
                              component.l_pre =
                                component.poloidal_pre / load_height
                                * displacement_to_love_scale;
                              component.l_post =
                                component.poloidal_post / load_height
                                * displacement_to_love_scale;
                              component.degree_horizontal_l_rms_pre =
                                horizontal_love_rms_by_degree[1];
                              component.degree_horizontal_l_rms_post =
                                projected_horizontal_love_rms_by_degree[1];
                              projection_components.push_back(component);
                            }
                        }

                      std::ofstream unified_output(output_directory +
                                                   "surface_love_number_coefficients" +
                                                   timestep_suffix);
                      unified_output << "# degree order h_cos h_sin k_cos k_sin l_cos l_sin geoid_cos geoid_sin surface_mass_potential_cos surface_mass_potential_sin tangential_displacement_cos tangential_displacement_sin radial_displacement_cos radial_displacement_sin h_from_radial_displacement_cos h_from_radial_displacement_sin degree_horizontal_l_rms";
                      if (degree_one_mass_moment_projection_prototype)
                        unified_output << " degree1_phi_over_g_total_pre_m degree1_phi_over_g_total_post_m degree1_projection_radial_correction_m degree1_projection_poloidal_correction_m projected_tangential_displacement_cos projected_tangential_displacement_sin projected_radial_displacement_cos projected_radial_displacement_sin projected_h_from_radial_displacement_cos projected_h_from_radial_displacement_sin projected_l_cos projected_l_sin projected_degree_horizontal_l_rms";
                      unified_output << "\n";
                      unified_output << "# field: load Love numbers plus raw surface spherical-harmonic coefficients\n";
                      unified_output << "# load_height_m: " << std::setprecision(16) << load_height << "\n";
                      unified_output << "# load_density_kg_m3: " << std::setprecision(16) << load_density << "\n";
                      unified_output << "# target_load_degree: " << load_degree << "\n";
                      unified_output << "# target_load_order: " << load_order << "\n";
                      unified_output << "# initial_elastic_displacement_time_s: " << std::setprecision(16) << initial_elastic_displacement_time << "\n";
                      unified_output << "# surface_radius_m: " << std::setprecision(16) << surface_radius << "\n";
                      unified_output << "# surface_gravity_m_s2: " << std::setprecision(16) << surface_gravity << "\n";
                      unified_output << "# h_lm = (surface_mass_potential_lm/load_geoid_scale_l - target_delta_lm) * displacement_to_love_scale_l\n";
                      unified_output << "# h_from_radial_displacement_lm = radial_displacement_lm/load_height * displacement_to_love_scale_l\n";
                      unified_output << "# k_lm = geoid_lm/load_geoid_scale_l - target_delta_lm\n";
                      unified_output << "# l_lm = tangential_displacement_lm/load_height * displacement_to_love_scale_l\n";
                      unified_output << "# displacement_to_love_scale_l = (2*l+1)*g/(4*pi*G*rho_load*R_surface)\n";
                      unified_output << "# geoid: total geoid-anomaly coefficient from the geoid postprocessor, m\n";
                      unified_output << "# surface_mass_potential: surface-topography mass-potential contribution from the geoid postprocessor, m\n";
                      unified_output << "# tangential_displacement: cumulative poloidal tangential displacement coefficient V_lm, m\n";
                      unified_output << "# radial_displacement: cumulative radial displacement coefficient U_lm, m\n";
                      unified_output << "# degree_horizontal_l_rms: sqrt(sum_m(l_cos^2+l_sin^2)) over all coefficients for the same degree\n";
                      unified_output << "# degree_1_mass_moment_projection_prototype: "
                                     << (degree_one_mass_moment_projection_prototype ? "true" : "false")
                                     << "\n";
                      unified_output << "# degree_1_mass_moment_projection_active_l1m0: "
                                     << (degree_one_projection_enabled ? "true" : "false")
                                     << "\n";
                      unified_output << "# degree_1_mass_moment_projection_poloidal_ratio: "
                                     << std::setprecision(16)
                                     << degree_one_mass_moment_projection_poloidal_ratio
                                     << "\n";
                      unified_output << "# degree_1_mass_moment_projection_phi_over_g_sensitivity_m_per_m: "
                                     << std::setprecision(16)
                                     << degree_one_projection_phi_sensitivity
                                     << "\n";
                      unified_output << "# degree1_phi_over_g_* columns are pre-reference-frame-cancellation Phi/g height coefficients in meters.\n";
                      unified_output << "# projection formula for l1m0: delta_U10 = -Phi_10_total_pre/(d(Phi_10/g)/dU10), delta_V10 = ratio*delta_U10.\n";
                      unified_output << "# degree_1_displacement_reference_frame: "
                                     << degree_one_displacement_reference_frame
                                     << "\n";
                      if (self_gravity != nullptr)
                        {
                          const Tensor<1,dim> total_cm =
                            self_gravity->get_cm_displacement_increment();
                          const Tensor<1,dim> deformation_cm =
                            self_gravity->get_deformation_cm_displacement_increment();
                          const Tensor<1,dim> external_load_cm =
                            self_gravity->get_external_load_cm_displacement_increment();
                          const Tensor<1,dim> surface_deformation_cm =
                            self_gravity->get_surface_deformation_cm_displacement_increment();
                          const Tensor<1,dim> cmb_deformation_cm =
                            self_gravity->get_cmb_deformation_cm_displacement_increment();
                          unified_output << "# total_cm_displacement_m: "
                                         << std::setprecision(16)
                                         << total_cm[0] << ' '
                                         << total_cm[1] << ' '
                                         << total_cm[2] << "\n";
                          unified_output << "# deformation_cm_displacement_m: "
                                         << std::setprecision(16)
                                         << deformation_cm[0] << ' '
                                         << deformation_cm[1] << ' '
                                         << deformation_cm[2] << "\n";
                          unified_output << "# external_load_cm_displacement_m: "
                                         << std::setprecision(16)
                                         << external_load_cm[0] << ' '
                                         << external_load_cm[1] << ' '
                                         << external_load_cm[2] << "\n";
                          unified_output << "# surface_deformation_cm_displacement_m: "
                                         << std::setprecision(16)
                                         << surface_deformation_cm[0] << ' '
                                         << surface_deformation_cm[1] << ' '
                                         << surface_deformation_cm[2] << "\n";
                          unified_output << "# cmb_deformation_cm_displacement_m: "
                                         << std::setprecision(16)
                                         << cmb_deformation_cm[0] << ' '
                                         << cmb_deformation_cm[1] << ' '
                                         << cmb_deformation_cm[2] << "\n";
                        }
                      if (degree_one_mass_moment_projection_prototype)
                        {
                          std::ofstream projection_output(
                            output_directory
                            + "degree_1_mass_moment_projection_diagnostic"
                            + timestep_suffix);
                          projection_output
                              << "# component phi_over_g_external_load_m phi_over_g_surface_deformation_m phi_over_g_cmb_deformation_m phi_over_g_volume_m phi_over_g_total_pre_m phi_over_g_total_post_m radial_correction_m poloidal_correction_m radial_displacement_pre_m radial_displacement_post_m poloidal_displacement_pre_m poloidal_displacement_post_m h_from_radial_displacement_pre h_from_radial_displacement_post l_pre l_post degree_horizontal_l_rms_pre degree_horizontal_l_rms_post\n";
                          projection_output
                              << "# degree_1_mass_moment_projection_active_l1m0: "
                              << (degree_one_projection_enabled ? "true" : "false")
                              << "\n";
                          projection_output
                              << "# projection formula: delta_U10 = -Phi_10_total_pre/S, delta_V10 = ratio*delta_U10\n";
                          projection_output
                              << "# S = 4*pi*G*Delta_rho_surface*R/(3*g) = "
                              << std::setprecision(16)
                              << degree_one_projection_phi_sensitivity
                              << " m Phi_over_g per m U10\n";
                          projection_output
                              << "# poloidal_ratio: "
                              << std::setprecision(16)
                              << degree_one_mass_moment_projection_poloidal_ratio
                              << "\n";
                          projection_output
                              << "# time: " << std::setprecision(16)
                              << this->get_time() << "\n";
                          projection_output
                              << "# timestep: " << this->get_timestep_number()
                              << "\n";

                          for (const DegreeOneProjectionComponent &component :
                               projection_components)
                            {
                              projection_output << component.name << ' '
                                                << std::setprecision(16)
                                                << component.external_load << ' '
                                                << component.surface_deformation << ' '
                                                << component.cmb_deformation << ' '
                                                << component.volume << ' '
                                                << component.total_pre << ' '
                                                << component.total_post << ' '
                                                << component.correction_radial << ' '
                                                << component.correction_poloidal << ' '
                                                << component.radial_pre << ' '
                                                << component.radial_post << ' '
                                                << component.poloidal_pre << ' '
                                                << component.poloidal_post << ' '
                                                << component.h_pre << ' '
                                                << component.h_post << ' '
                                                << component.l_pre << ' '
                                                << component.l_post << ' '
                                                << component.degree_horizontal_l_rms_pre << ' '
                                                << component.degree_horizontal_l_rms_post << '\n';
                            }
                        }
                      unified_output << "# time: " << std::setprecision(16) << this->get_time() << "\n";
                      unified_output << "# timestep: " << this->get_timestep_number() << "\n";

                      const double G = aspect::constants::big_g;

                      coefficient_index = 0;
                      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                        for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                          {
                            const std::pair<double,double> self_gravity_surface =
                              surface_mass_potential_coefficient(degree, order);

                            const std::pair<double,double> self_gravity_cmb =
                              cmb_mass_potential_coefficient(degree, order);

                            const std::pair<double,double> self_gravity_tidal =
                              tidal_surface_potential_coefficient(degree, order);

                            const std::pair<double,double> rotational_potential =
                              rotational_surface_potential_coefficient(degree,
                                                                       order);

                            const std::pair<double,double> self_gravity_reference_frame =
                              reference_frame_surface_potential_coefficient(degree, order);

                            const double coecos_density_anomaly =
                              (4.0 * numbers::PI * G / (surface_gravity * (2.0 * degree + 1.0)))
                              * self_gravity_vector_coefficient(
                                SH_density_coes,
                                degree,
                                order,
                                false);
                            const double coesin_density_anomaly =
                              (4.0 * numbers::PI * G / (surface_gravity * (2.0 * degree + 1.0)))
                              * self_gravity_vector_coefficient(
                                SH_density_coes,
                                degree,
                                order,
                                true);

                            const double coecos_surface_topo =
                              (use_boundary_potential
                               ? self_gravity_surface.first
                               : (4.0 * numbers::PI * G / (surface_gravity * (2.0 * degree + 1.0)))
                               * surface_delta_rho
                               * self_gravity_vector_coefficient(
                                 SH_surface_topo_coes.second,
                                 degree,
                                 order,
                                 false)
                               * surface_radius);
                            const double coesin_surface_topo =
                              (use_boundary_potential
                               ? self_gravity_surface.second
                               : (4.0 * numbers::PI * G / (surface_gravity * (2.0 * degree + 1.0)))
                               * surface_delta_rho
                               * self_gravity_vector_coefficient(
                                 SH_surface_topo_coes.second,
                                 degree,
                                 order,
                                 true)
                               * surface_radius);

                            const double coecos_CMB_topo =
                              (use_boundary_potential
                               ? self_gravity_cmb.first
                               : (4.0 * numbers::PI * G / (surface_gravity * (2.0 * degree + 1.0)))
                               * CMB_delta_rho
                               * self_gravity_vector_coefficient(
                                 SH_CMB_topo_coes.second,
                                 degree,
                                 order,
                                 false)
                               * inner_radius
#if DEAL_II_VERSION_GTE(9,6,0)
                               * Utilities::pow(inner_radius / surface_radius, degree + 1));
#else
                               * std::pow(inner_radius / surface_radius, degree + 1));
#endif
                            const double coesin_CMB_topo =
                              (use_boundary_potential
                               ? self_gravity_cmb.second
                               : (4.0 * numbers::PI * G / (surface_gravity * (2.0 * degree + 1.0)))
                               * CMB_delta_rho
                               * self_gravity_vector_coefficient(
                                 SH_CMB_topo_coes.second,
                                 degree,
                                 order,
                                 true)
                               * inner_radius
#if DEAL_II_VERSION_GTE(9,6,0)
                               * Utilities::pow(inner_radius / surface_radius, degree + 1));
#else
                               * std::pow(inner_radius / surface_radius, degree + 1));
#endif

                            const double coecos_tidal =
                              (use_boundary_potential
                               ? self_gravity_tidal.first
                               : 0.0);
                            const double coesin_tidal =
                              (use_boundary_potential
                               ? self_gravity_tidal.second
                               : 0.0);
                            const double coecos_rotational =
                              (use_boundary_potential
                               ? rotational_potential.first
                               : 0.0);
                            const double coesin_rotational =
                              (use_boundary_potential
                               ? rotational_potential.second
                               : 0.0);
                            const double coecos_reference_frame =
                              (use_boundary_potential
                               ? self_gravity_reference_frame.first
                               : 0.0);
                            const double coesin_reference_frame =
                              (use_boundary_potential
                               ? self_gravity_reference_frame.second
                               : 0.0);

                            const std::pair<double,double> geoid_coefficient =
                              std::make_pair(coecos_density_anomaly
                                             + coecos_surface_topo
                                             + coecos_CMB_topo
                                             + coecos_tidal
                                             + coecos_rotational
                                             + coecos_reference_frame,
                                             coesin_density_anomaly
                                             + coesin_surface_topo
                                             + coesin_CMB_topo
                                             + coesin_tidal
                                             + coesin_rotational
                                             + coesin_reference_frame);

                            const std::pair<double,double> surface_coefficient =
                              std::make_pair(coecos_surface_topo, coesin_surface_topo);

                            const double load_geoid_scale =
                              4.0 * numbers::PI * constants::big_g
                              * load_density * load_height * surface_radius
                              / (surface_gravity * (2.0 * degree + 1.0));
                            const bool is_target =
                              (degree == load_degree && order == load_order);
                            const double target_cos_delta = (is_target ? 1.0 : 0.0);
                            const double displacement_to_love_scale =
                              (2.0 * degree + 1.0) * surface_gravity
                              / (4.0 * numbers::PI * constants::big_g
                                 * load_density * surface_radius);
                            const double k_cos =
                              geoid_coefficient.first / load_geoid_scale - target_cos_delta;
                            const double k_sin =
                              geoid_coefficient.second / load_geoid_scale;
                            const std::pair<double,double> tangential_displacement =
                              adjusted_tangential_displacement(degree,
                                                               order,
                                                               coefficient_index);
                            const double tangential_displacement_cos =
                              tangential_displacement.first;
                            const double tangential_displacement_sin =
                              tangential_displacement.second;
                            const std::pair<double,double> radial_displacement =
                              adjusted_radial_displacement(degree,
                                                           order,
                                                           coefficient_index);
                            const double radial_displacement_cos =
                              radial_displacement.first;
                            const double radial_displacement_sin =
                              radial_displacement.second;
                            const std::pair<double,double> projected_tangential =
                              projected_tangential_displacement(degree,
                                                                order,
                                                                coefficient_index);
                            const std::pair<double,double> projected_radial =
                              projected_radial_displacement(degree,
                                                            order,
                                                            coefficient_index);
                            double h_cos =
                              (surface_coefficient.first / load_geoid_scale - target_cos_delta)
                              * displacement_to_love_scale;
                            double h_sin =
                              surface_coefficient.second / load_geoid_scale
                              * displacement_to_love_scale;
                            const double l_cos =
                              tangential_displacement_cos / load_height
                              * displacement_to_love_scale;
                            const double l_sin =
                              tangential_displacement_sin / load_height
                              * displacement_to_love_scale;
                            const double h_from_radial_displacement_cos =
                              radial_displacement_cos / load_height
                              * displacement_to_love_scale;
                            const double h_from_radial_displacement_sin =
                              radial_displacement_sin / load_height
                              * displacement_to_love_scale;
                            const double projected_l_cos =
                              projected_tangential.first / load_height
                              * displacement_to_love_scale;
                            const double projected_l_sin =
                              projected_tangential.second / load_height
                              * displacement_to_love_scale;
                            const double projected_h_from_radial_displacement_cos =
                              projected_radial.first / load_height
                              * displacement_to_love_scale;
                            const double projected_h_from_radial_displacement_sin =
                              projected_radial.second / load_height
                              * displacement_to_love_scale;
                            if (degree_one_displacement_reference_frame == "citcomsve-deformation-cm"
                                && degree == 1)
                              {
                                h_cos = h_from_radial_displacement_cos;
                                h_sin = h_from_radial_displacement_sin;
                              }

                            double row_phi_total_pre = std::numeric_limits<double>::quiet_NaN();
                            double row_phi_total_post = std::numeric_limits<double>::quiet_NaN();
                            double row_projection_radial_correction = std::numeric_limits<double>::quiet_NaN();
                            double row_projection_poloidal_correction = std::numeric_limits<double>::quiet_NaN();
                            if (degree_one_projection_enabled
                                && degree == 1
                                && order == 0)
                              {
                                row_phi_total_pre =
                                  degree_one_projection_phi_total_pre;
                                row_projection_radial_correction =
                                  degree_one_projection_radial_correction;
                                row_projection_poloidal_correction =
                                  degree_one_projection_poloidal_correction;
                                row_phi_total_post =
                                  degree_one_projection_phi_total_pre
                                  + degree_one_projection_phi_sensitivity
                                  * degree_one_projection_radial_correction;
                              }

                            unified_output << degree << ' '
                                           << order << ' '
                                           << std::setprecision(16) << h_cos << ' '
                                           << std::setprecision(16) << h_sin << ' '
                                           << std::setprecision(16) << k_cos << ' '
                                           << std::setprecision(16) << k_sin << ' '
                                           << std::setprecision(16) << l_cos << ' '
                                           << std::setprecision(16) << l_sin << ' '
                                           << std::setprecision(16) << geoid_coefficient.first << ' '
                                           << std::setprecision(16) << geoid_coefficient.second << ' '
                                           << std::setprecision(16) << surface_coefficient.first << ' '
                                           << std::setprecision(16) << surface_coefficient.second << ' '
                                           << std::setprecision(16) << tangential_displacement_cos << ' '
                                           << std::setprecision(16) << tangential_displacement_sin << ' '
                                           << std::setprecision(16) << radial_displacement_cos << ' '
                                           << std::setprecision(16) << radial_displacement_sin << ' '
                                           << std::setprecision(16) << h_from_radial_displacement_cos << ' '
                                           << std::setprecision(16) << h_from_radial_displacement_sin << ' '
                                           << std::setprecision(16) << horizontal_love_rms_by_degree[degree];
                            if (degree_one_mass_moment_projection_prototype)
                              unified_output << ' '
                                             << std::setprecision(16) << row_phi_total_pre << ' '
                                             << std::setprecision(16) << row_phi_total_post << ' '
                                             << std::setprecision(16) << row_projection_radial_correction << ' '
                                             << std::setprecision(16) << row_projection_poloidal_correction << ' '
                                             << std::setprecision(16) << projected_tangential.first << ' '
                                             << std::setprecision(16) << projected_tangential.second << ' '
                                             << std::setprecision(16) << projected_radial.first << ' '
                                             << std::setprecision(16) << projected_radial.second << ' '
                                             << std::setprecision(16) << projected_h_from_radial_displacement_cos << ' '
                                             << std::setprecision(16) << projected_h_from_radial_displacement_sin << ' '
                                             << std::setprecision(16) << projected_l_cos << ' '
                                             << std::setprecision(16) << projected_l_sin << ' '
                                             << std::setprecision(16) << projected_horizontal_love_rms_by_degree[degree];
                            unified_output << '\n';
                          }
                    }
                }

              last_text_output_time = this->get_time();
            }

          std::ostringstream output;
          output.precision(4);
          output << std::scientific
                 << "tracked " << n_coefficients << " coefficients";

          return std::make_pair("Surface love numbers:",
                                output.str());
        }
    }



    template <int dim>
    std::list<std::string>
    SurfaceLoveNumbers<dim>::required_other_postprocessors() const
    {
      std::list<std::string> deps;
      return deps;
    }



    template <int dim>
    void
    SurfaceLoveNumbers<dim>::initialize_simulator (const Simulator<dim> &simulator)
    {
      SimulatorAccess<dim>::initialize_simulator(simulator);
      self_gravity_helper.initialize_simulator(simulator);
    }


    template <int dim>
    void
    SurfaceLoveNumbers<dim>::initialize ()
    {
      const auto &traction_manager = this->get_boundary_traction_manager();
      const bool use_self_gravity =
        traction_manager.template has_matching_active_plugin<
        PotentialFeedback::SelfGravitation<dim>>();
      bool use_potential_feedback = false;
      if (traction_manager.template has_matching_active_plugin<
          BoundaryTraction::PotentialFeedbackTraction<dim>>())
        {
          const auto &pf = traction_manager.template get_matching_active_plugin<
                           BoundaryTraction::PotentialFeedbackTraction<dim>>();
          use_potential_feedback = pf.has_self_gravity_feedback();
        }

      if (!use_self_gravity && !use_potential_feedback)
        self_gravity_helper.initialize();
    }


    template <int dim>
    void
    SurfaceLoveNumbers<dim>::declare_parameters (ParameterHandler &prm)
    {
      PotentialFeedback::SelfGravitation<dim>::declare_parameters(prm);
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Surface love numbers");
        {
          prm.declare_entry("Maximum degree", "32",
                            Patterns::Integer(0),
                            "Maximum spherical-harmonic degree for the tangential "
                            "surface-displacement projection.");
          prm.declare_entry("Minimum degree", "0",
                            Patterns::Integer(0),
                            "Minimum spherical-harmonic degree for the tangential "
                            "surface-displacement projection.");
          prm.declare_entry("Output coefficients", "true",
                            Patterns::Bool(),
                            "Whether to write one text file per output time "
                            "with normalized h, k, and l load Love numbers "
                            "followed by the raw geoid, surface mass-potential, "
                            "and cumulative tangential displacement coefficients "
                            "used to compute them.");
          prm.declare_entry("Initial elastic displacement time", "0",
                            Patterns::Double(0),
                            "Time interval used to convert the timestep-zero "
                            "instantaneous elastic tangential velocity to an "
                            "initial tangential displacement. Units follow the "
                            "model time setting.");
          prm.declare_entry("Degree 1 displacement reference frame", "solution",
                            Patterns::Selection("solution|nullspace-restored|citcomsve-deformation-cm"),
                            "Reference frame used for degree-1 tangential "
                            "displacement diagnostics. The default 'solution' "
                            "uses the ASPECT velocity after nullspace removal. "
                            "The 'nullspace-restored' option adds the uniform "
                            "translation removed by ASPECT's net-translation "
                            "nullspace removal back to degree-1 poloidal "
                            "displacement coefficients before computing l; "
                            "this is intended only as a diagnostic. The "
                            "'citcomsve-deformation-cm' option subtracts the "
                            "deformation-only center-of-mass increment diagnosed "
                            "by the self-gravity feedback from degree-1 radial "
                            "and poloidal displacement coefficients, matching "
                            "CitcomSVE's shift_to_CM output-frame operation.");
          prm.declare_entry("Degree 1 center of mass displacement scale", "1.0",
                            Patterns::Double(),
                            "Scale factor applied to the diagnosed center-of-mass "
                            "displacement when using the 'citcomsve-deformation-cm' "
                            "degree-1 displacement reference frame. This diagnostic "
                            "parameter is ignored for other reference frames.");
          prm.declare_entry("Degree 1 mass moment projection prototype", "false",
                            Patterns::Bool(),
                            "Whether to run a default-off l=1,m=0 diagnostic "
                            "prototype that projects the degree-1 radial and "
                            "poloidal displacement coefficients toward zero "
                            "pre-reference-frame-cancellation Phi_10/g. This "
                            "does not modify the Stokes solve or production "
                            "degree >= 2 Love-number diagnostics.");
          prm.declare_entry("Degree 1 boundary traction RHS diagnostic", "false",
                            Patterns::Bool(),
                            "Whether to write a default-off diagnostic that "
                            "projects the actual top and bottom boundary "
                            "traction returned by the boundary-traction "
                            "manager onto the normalized Y10 spherical "
                            "harmonic. This is intended to compare ASPECT's "
                            "solve-stage Neumann RHS against CitcomSVE's "
                            "post-append load[1]/load[3] coefficients.");
          prm.declare_entry("Degree 1 mass moment projection poloidal ratio", "-1.0",
                            Patterns::Double(),
                            "Diagnostic ratio dV_10/dU_10 used by the degree-1 "
                            "mass-moment projection prototype. The default "
                            "uses opposite radial and poloidal coefficient "
                            "corrections and is ignored unless the prototype "
                            "is enabled.");
          prm.declare_entry("Time between text output", "0",
                            Patterns::Double(0),
                            "Time interval between text outputs. A value of zero "
                            "disables time-based output control.");
          prm.declare_entry("Time steps between text output", "1",
                            Patterns::Integer(0),
                            "Number of time steps between text outputs. A value "
                            "of zero disables timestep-based output control.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    SurfaceLoveNumbers<dim>::parse_parameters (ParameterHandler &prm)
    {
      self_gravity_helper.parse_parameters(prm);
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Surface love numbers");
        {
          max_degree = prm.get_integer("Maximum degree");
          min_degree = prm.get_integer("Minimum degree");
          output_coefficients = prm.get_bool("Output coefficients");
          initial_elastic_displacement_time =
            prm.get_double("Initial elastic displacement time");
          degree_one_displacement_reference_frame =
            prm.get("Degree 1 displacement reference frame");
          degree_one_cm_displacement_scale =
            prm.get_double("Degree 1 center of mass displacement scale");
          degree_one_mass_moment_projection_prototype =
            prm.get_bool("Degree 1 mass moment projection prototype");
          degree_one_boundary_traction_rhs_diagnostic =
            prm.get_bool("Degree 1 boundary traction RHS diagnostic");
          degree_one_mass_moment_projection_poloidal_ratio =
            prm.get_double("Degree 1 mass moment projection poloidal ratio");
          time_between_text_output = prm.get_double("Time between text output");
          time_steps_between_text_output = prm.get_integer("Time steps between text output");
          if (this->convert_output_to_years())
            {
              initial_elastic_displacement_time *= constants::year_in_seconds;
              time_between_text_output *= constants::year_in_seconds;
            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Spherical harmonic load");
        {
          load_degree = prm.get_integer("Harmonic degree");
          load_order = prm.get_integer("Harmonic order");
          load_height = prm.get_double("Load height");
          load_density = prm.get_double("Load density");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(min_degree <= max_degree,
                  ExcMessage("Minimum degree must be smaller than or equal to maximum degree."));
      AssertThrow(load_order <= load_degree,
                  ExcMessage("Spherical harmonic load order must be smaller than or equal to load degree."));
      AssertThrow(load_height > 0.0,
                  ExcMessage("Surface love numbers require Spherical harmonic load/Load height to be positive."));
      AssertThrow(load_density > 0.0,
                  ExcMessage("Surface love numbers require Spherical harmonic load/Load density to be positive."));
    }



    template <int dim>
    template <class Archive>
    void
    SurfaceLoveNumbers<dim>::serialize (Archive &ar, const unsigned int)
    {
      ar &last_text_output_time
      & displacement_coecos
      & displacement_coesin
      & radial_displacement_coecos
      & radial_displacement_coesin;
    }



    template <int dim>
    void
    SurfaceLoveNumbers<dim>::save (std::map<std::string, std::string> &status_strings) const
    {
      std::ostringstream os;
      {
        aspect::oarchive oa (os);
        oa << (*this);
      }

      status_strings["SurfaceLoveNumbers"] = os.str();
    }



    template <int dim>
    void
    SurfaceLoveNumbers<dim>::load (const std::map<std::string, std::string> &status_strings)
    {
      if (status_strings.find("SurfaceLoveNumbers") != status_strings.end())
        {
          std::istringstream is (status_strings.find("SurfaceLoveNumbers")->second);
          aspect::iarchive ia (is);
          ia >> (*this);
        }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(SurfaceLoveNumbers,
                                  "surface love numbers",
                                  "A postprocessor that writes the surface "
                                  "spherical-harmonic coefficients needed to "
                                  "compute load Love numbers. It combines "
                                  "the geoid postprocessor's geoid and surface "
                                  "mass-potential coefficients with the "
                                  "cumulative tangential displacement "
                                  "coefficients.")
  }
}
