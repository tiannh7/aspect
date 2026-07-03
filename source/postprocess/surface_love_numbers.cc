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
#include <aspect/postprocess/geoid.h>
#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iomanip>
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

          double timestep = this->get_timestep();
          if (this->get_timestep_number() == 0 && initial_elastic_displacement_time > 0.0)
            timestep = initial_elastic_displacement_time;
          std::vector<double> local_increment_cos(n_coefficients, 0.0);
          std::vector<double> local_increment_sin(n_coefficients, 0.0);

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
                                  const double normalization =
                                    (degree > 0 ? static_cast<double>(degree * (degree + 1)) : 1.0);

                                  local_increment_cos[coefficient_index] +=
                                    (tangential_displacement_3d * gradients.first) * area_weight / normalization;
                                  local_increment_sin[coefficient_index] +=
                                    (tangential_displacement_3d * gradients.second) * area_weight / normalization;
                                }
                          }
                      }
            }

          std::vector<double> global_increment_cos(n_coefficients, 0.0);
          std::vector<double> global_increment_sin(n_coefficients, 0.0);
          Utilities::MPI::sum(local_increment_cos,
                              this->get_mpi_communicator(),
                              global_increment_cos);
          Utilities::MPI::sum(local_increment_sin,
                              this->get_mpi_communicator(),
                              global_increment_sin);

          for (unsigned int i=0; i<n_coefficients; ++i)
            {
              displacement_coecos[i] += global_increment_cos[i];
              displacement_coesin[i] += global_increment_sin[i];
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

              if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
                {
                  const std::string timestep_suffix =
                    "." + Utilities::int_to_string(this->get_timestep_number(), 5);
                  const std::string output_directory =
                    this->get_output_directory() + "surface_love_numbers/";
                  const Point<dim> surface_point =
                    this->get_geometry_model().representative_point(1.0);
                  const double surface_radius = surface_point.norm();
                  const double surface_gravity =
                    this->get_gravity_model().gravity_vector(surface_point).norm();

                  std::ofstream displacement_output(output_directory +
                                                    "surface_tangential_displacement_SH_coefficients" +
                                                    timestep_suffix);
                  displacement_output << "# degree order cosine_coefficient sine_coefficient\n";
                  displacement_output << "# field: cumulative poloidal tangential displacement coefficient V_lm, m\n";
                  displacement_output << "# projection: integral(u_t dot grad_s Y_lm) / l(l+1)\n";
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

                  if (output_coefficients)
                    {
                      const Geoid<dim> &geoid_postprocessor =
                        this->get_postprocess_manager().template get_matching_active_plugin<Geoid<dim>>();

                      std::ofstream unified_output(output_directory +
                                                   "surface_love_number_coefficients" +
                                                   timestep_suffix);
                      unified_output << "# degree order h_cos h_sin k_cos k_sin l_cos l_sin geoid_cos geoid_sin surface_mass_potential_cos surface_mass_potential_sin tangential_displacement_cos tangential_displacement_sin\n";
                      unified_output << "# field: load Love numbers plus raw surface spherical-harmonic coefficients\n";
                      unified_output << "# load_height_m: " << std::setprecision(16) << load_height << "\n";
                      unified_output << "# load_density_kg_m3: " << std::setprecision(16) << load_density << "\n";
                      unified_output << "# target_load_degree: " << load_degree << "\n";
                      unified_output << "# target_load_order: " << load_order << "\n";
                      unified_output << "# initial_elastic_displacement_time_s: " << std::setprecision(16) << initial_elastic_displacement_time << "\n";
                      unified_output << "# surface_radius_m: " << std::setprecision(16) << surface_radius << "\n";
                      unified_output << "# surface_gravity_m_s2: " << std::setprecision(16) << surface_gravity << "\n";
                      unified_output << "# h_lm = surface_mass_potential_lm/load_geoid_scale_l - target_delta_lm\n";
                      unified_output << "# k_lm = geoid_lm/load_geoid_scale_l - target_delta_lm\n";
                      unified_output << "# l_lm = tangential_displacement_lm/load_height * displacement_to_love_scale_l\n";
                      unified_output << "# displacement_to_love_scale_l = (2*l+1)*g/(4*pi*G*rho_load*R_surface)\n";
                      unified_output << "# geoid: total geoid-anomaly coefficient from the geoid postprocessor, m\n";
                      unified_output << "# surface_mass_potential: surface-topography mass-potential contribution from the geoid postprocessor, m\n";
                      unified_output << "# tangential_displacement: cumulative poloidal tangential displacement coefficient V_lm, m\n";
                      unified_output << "# time: " << std::setprecision(16) << this->get_time() << "\n";
                      unified_output << "# timestep: " << this->get_timestep_number() << "\n";

                      coefficient_index = 0;
                      for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
                        for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                          {
                            const std::pair<double,double> geoid_coefficient =
                              geoid_postprocessor.geoid_coefficient(degree, order);
                            const std::pair<double,double> surface_coefficient =
                              geoid_postprocessor.surface_topography_contribution_coefficient(degree, order);

                            const double load_geoid_scale =
                              4.0 * numbers::PI * constants::big_g
                              * load_density * load_height * surface_radius
                              / (surface_gravity * (2.0 * degree + 1.0));
                            const bool is_target =
                              (degree == load_degree && order == load_order);
                            const double target_cos_delta = (is_target ? 1.0 : 0.0);
                            const double h_cos =
                              surface_coefficient.first / load_geoid_scale - target_cos_delta;
                            const double h_sin =
                              surface_coefficient.second / load_geoid_scale;
                            const double k_cos =
                              geoid_coefficient.first / load_geoid_scale - target_cos_delta;
                            const double k_sin =
                              geoid_coefficient.second / load_geoid_scale;
                            const double displacement_to_love_scale =
                              (2.0 * degree + 1.0) * surface_gravity
                              / (4.0 * numbers::PI * constants::big_g
                                 * load_density * surface_radius);
                            const double l_cos =
                              displacement_coecos[coefficient_index] / load_height
                              * displacement_to_love_scale;
                            const double l_sin =
                              displacement_coesin[coefficient_index] / load_height
                              * displacement_to_love_scale;

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
                                           << std::setprecision(16) << displacement_coecos[coefficient_index] << ' '
                                           << std::setprecision(16) << displacement_coesin[coefficient_index] << '\n';
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

      if (output_coefficients)
        deps.emplace_back("geoid");

      return deps;
    }



    template <int dim>
    void
    SurfaceLoveNumbers<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Surface love numbers");
        {
          prm.declare_entry("Maximum degree", "32",
                            Patterns::Integer(0),
                            "Maximum spherical-harmonic degree for the tangential "
                            "surface-displacement projection.");
          prm.declare_entry("Minimum degree", "1",
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
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Surface love numbers");
        {
          max_degree = prm.get_integer("Maximum degree");
          min_degree = prm.get_integer("Minimum degree");
          output_coefficients = prm.get_bool("Output coefficients");
          initial_elastic_displacement_time =
            prm.get_double("Initial elastic displacement time");
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
      & displacement_coesin;
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
