/*
  Copyright (C) 2011 - 2025 by the authors of the ASPECT code.

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


#include <aspect/postprocess/surface_stress_statistics.h>
#include <aspect/material_model/rheology/elasticity.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <map>
#include <iomanip>
#include <array>
#include <tuple>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    SurfaceStressStatistics<dim>::execute (TableHandler &statistics)
    {
      const bool output_spherical = this->get_postprocess_manager().get_output_in_spherical_coordinates();

      // Component names depending on coordinate system
      const bool is_spherical_like = (this->get_geometry_model().natural_coordinate_system() != Utilities::Coordinates::cartesian);
      const std::vector<std::string> stress_component_names_2d = (output_spherical || is_spherical_like)
                                                                 ? std::vector<std::string> {"rr","pp","rp"}
                                                                 :
                                                                 std::vector<std::string> {"xx","yy","xy"};
      const std::vector<std::string> stress_component_names_3d = (output_spherical || is_spherical_like)
                                                                 ? std::vector<std::string> {"rr","tt","pp","rt","rp","tp"}
                                                                 :
                                                                 std::vector<std::string> {"xx","yy","zz","xy","xz","yz"};

      const Quadrature<dim-1> &quadrature_formula = this->introspection().face_quadratures.velocities;

      FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                        this->get_fe(),
                                        quadrature_formula,
                                        update_values |
                                        update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);

      const unsigned int n_q_points = fe_face_values.n_quadrature_points;

      // Prepare containers for material inputs/outputs evaluated on faces
      std::vector<double> pressure_values(n_q_points);
      std::vector<double> temperature_values(n_q_points);
      std::vector<Tensor<2,dim>> velocity_gradients(n_q_points);

      MaterialModel::MaterialModelInputs<dim> in(n_q_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(n_q_points, this->n_compositional_fields());
      in.requested_properties = MaterialModel::MaterialProperties::viscosity | MaterialModel::MaterialProperties::additional_outputs;
      this->get_material_model().create_additional_named_outputs(out);

      // Maps keyed by boundary id to arrays over tensor components.
      std::map<types::boundary_id, std::vector<double>> local_min_stress;
      std::map<types::boundary_id, std::vector<double>> local_max_stress;
      std::map<types::boundary_id, std::vector<double>> local_stress_integral;

      std::map<types::boundary_id, std::vector<double>> local_min_shear_stress;
      std::map<types::boundary_id, std::vector<double>> local_max_shear_stress;
      std::map<types::boundary_id, std::vector<double>> local_shear_stress_integral;
      std::map<types::boundary_id, double> local_boundary_area;

      const types::boundary_id top_boundary_id = (dim == 3
                                                  ? this->get_geometry_model().translate_symbolic_boundary_name_to_id("top")
                                                  : numbers::invalid_boundary_id);
      const unsigned int stress_sh_max_degree = 20;
      const unsigned int n_stress_sh_coefficients = (stress_sh_max_degree + 1) * (stress_sh_max_degree + 2) / 2;

      const unsigned int n_components = SymmetricTensor<2,dim>::n_independent_components;
      std::vector<std::vector<double>> local_surface_total_stress_sh_cos(n_components,
                                                                          std::vector<double>(n_stress_sh_coefficients, 0.0));
      std::vector<std::vector<double>> local_surface_total_stress_sh_sin(n_components,
                                                                          std::vector<double>(n_stress_sh_coefficients, 0.0));
      std::vector<std::vector<double>> local_surface_deviatoric_stress_sh_cos(n_components,
                                                                               std::vector<double>(n_stress_sh_coefficients, 0.0));
      std::vector<std::vector<double>> local_surface_deviatoric_stress_sh_sin(n_components,
                                                                               std::vector<double>(n_stress_sh_coefficients, 0.0));
      std::vector<double> local_surface_tangential_deviatoric_stress_sh_cos(n_stress_sh_coefficients, 0.0);
      std::vector<double> local_surface_tangential_deviatoric_stress_sh_sin(n_stress_sh_coefficients, 0.0);

      const std::set<types::boundary_id> boundary_indicators = this->get_geometry_model().get_used_boundary_indicators();
      for (const auto id : boundary_indicators)
        {
          local_min_stress[id] = std::vector<double>(n_components, std::numeric_limits<double>::max());
          local_max_stress[id] = std::vector<double>(n_components, std::numeric_limits<double>::lowest());
          local_stress_integral[id] = std::vector<double>(n_components, 0.0);

          local_min_shear_stress[id] = std::vector<double>(n_components, std::numeric_limits<double>::max());
          local_max_shear_stress[id] = std::vector<double>(n_components, std::numeric_limits<double>::lowest());
          local_shear_stress_integral[id] = std::vector<double>(n_components, 0.0);
          local_boundary_area[id] = 0.0;
        }

      // Loop over boundary faces
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                fe_face_values.reinit(cell, f);

                fe_face_values[this->introspection().extractors.pressure].get_function_values(this->get_solution(), pressure_values);
                fe_face_values[this->introspection().extractors.temperature].get_function_values(this->get_solution(), temperature_values);
                fe_face_values[this->introspection().extractors.velocities].get_function_gradients(this->get_solution(), velocity_gradients);

                // Fill material inputs for faces
                in.position = fe_face_values.get_quadrature_points();
                in.temperature = temperature_values;
                in.pressure = pressure_values;
                in.velocity = std::vector<Tensor<1,dim>>(n_q_points);

                // Get composition values directly into the correct structure [n_q_points][n_fields]
                for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                  {
                    std::vector<double> composition_values_c(n_q_points);
                    fe_face_values[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(), composition_values_c);
                    for (unsigned int q=0; q<n_q_points; ++q)
                      in.composition[q][c] = composition_values_c[q];
                  }

                for (unsigned int q=0; q<n_q_points; ++q)
                  in.strain_rate[q] = symmetrize(velocity_gradients[q]);

                this->get_material_model().evaluate(in, out);

                const types::boundary_id bid = cell->face(f)->boundary_id();
                for (unsigned int q=0; q<n_q_points; ++q)
                  {
                    SymmetricTensor<2,dim> deviatoric_stress;
                    if (this->get_parameters().enable_elasticity)
                      {
                        const std::shared_ptr<const MaterialModel::ElasticAdditionalOutputs<dim>> elastic_out =
                          out.template get_additional_output_object<MaterialModel::ElasticAdditionalOutputs<dim>>();
                        Assert(elastic_out != nullptr, ExcMessage("Elastic Additional Outputs are needed for surface stress statistics."));
                        deviatoric_stress = elastic_out->deviatoric_stress[q];
                      }
                    else
                      {
                        const double eta = out.viscosities[q];
                        const SymmetricTensor<2,dim> sr = in.strain_rate[q];
                        const SymmetricTensor<2,dim> dsr = (this->get_material_model().is_compressible()
                                                            ? sr - (1./3.) * trace(sr) * unit_symmetric_tensor<dim>()
                                                            : sr);
                        deviatoric_stress = 2.0 * eta * dsr;
                      }

                    // Total stress and shear/deviatoric stress use the geoscience
                    // sign convention, matching StressStatistics.
                    SymmetricTensor<2,dim> stress = in.pressure[q] * unit_symmetric_tensor<dim>();
                    stress -= deviatoric_stress;
                    SymmetricTensor<2,dim> shear_stress = -deviatoric_stress;
                    if (output_spherical || is_spherical_like)
                      {
                        stress = Utilities::Coordinates::cartesian_to_spherical_tensor(stress, in.position[q]);
                        shear_stress = - Utilities::Coordinates::cartesian_to_spherical_tensor(deviatoric_stress, in.position[q]);
                      }

                    if constexpr (dim == 3)
                      if (bid == top_boundary_id)
                        {
                          const std::array<double,dim> spherical_coordinates =
                            Utilities::Coordinates::cartesian_to_spherical_coordinates(in.position[q]);
                          const double radius = spherical_coordinates[0];
                          const double phi = spherical_coordinates[1];
                          const double theta = spherical_coordinates[2];
                          const double area_weight = fe_face_values.JxW(q) / (radius * radius);
                          const double tangential_deviatoric_stress =
                            0.5 * (shear_stress[1][1] + shear_stress[2][2]);

                          for (unsigned int degree = 0, coefficient_index = 0;
                               degree <= stress_sh_max_degree;
                               ++degree)
                            for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                              {
                                const std::pair<double,double> spherical_harmonic =
                                  Utilities::real_spherical_harmonic(degree, order, theta, phi);
                                local_surface_tangential_deviatoric_stress_sh_cos[coefficient_index] +=
                                  tangential_deviatoric_stress * spherical_harmonic.first * area_weight;
                                local_surface_tangential_deviatoric_stress_sh_sin[coefficient_index] +=
                                  tangential_deviatoric_stress * spherical_harmonic.second * area_weight;

                                for (unsigned int i=0; i<n_components; ++i)
                                  {
                                    const TableIndices<2> idx = SymmetricTensor<2,dim>::unrolled_to_component_indices(i);
                                    const double total_value = stress[idx[0]][idx[1]];
                                    const double deviatoric_value = shear_stress[idx[0]][idx[1]];
                                    local_surface_total_stress_sh_cos[i][coefficient_index] +=
                                      total_value * spherical_harmonic.first * area_weight;
                                    local_surface_total_stress_sh_sin[i][coefficient_index] +=
                                      total_value * spherical_harmonic.second * area_weight;
                                    local_surface_deviatoric_stress_sh_cos[i][coefficient_index] +=
                                      deviatoric_value * spherical_harmonic.first * area_weight;
                                    local_surface_deviatoric_stress_sh_sin[i][coefficient_index] +=
                                      deviatoric_value * spherical_harmonic.second * area_weight;
                                  }
                              }
                        }

                    for (unsigned int i=0; i<n_components; ++i)
                      {
                        const TableIndices<2> idx = SymmetricTensor<2,dim>::unrolled_to_component_indices(i);
                        const double stress_value = stress[idx[0]][idx[1]];
                        const double shear_stress_value = shear_stress[idx[0]][idx[1]];

                        local_min_stress[bid][i] = std::min(local_min_stress[bid][i], stress_value);
                        local_max_stress[bid][i] = std::max(local_max_stress[bid][i], stress_value);
                        local_stress_integral[bid][i] += stress_value * fe_face_values.JxW(q);

                        local_min_shear_stress[bid][i] = std::min(local_min_shear_stress[bid][i], shear_stress_value);
                        local_max_shear_stress[bid][i] = std::max(local_max_shear_stress[bid][i], shear_stress_value);
                        local_shear_stress_integral[bid][i] += shear_stress_value * fe_face_values.JxW(q);
                      }
                    local_boundary_area[bid] += fe_face_values.JxW(q);
                  }
              }

      // MPI reductions per boundary id
      std::map<types::boundary_id, std::vector<double>> global_min_stress;
      std::map<types::boundary_id, std::vector<double>> global_max_stress;
      std::map<types::boundary_id, std::vector<double>> global_avg_stress;

      std::map<types::boundary_id, std::vector<double>> global_min_shear_stress;
      std::map<types::boundary_id, std::vector<double>> global_max_shear_stress;
      std::map<types::boundary_id, std::vector<double>> global_avg_shear_stress;

      for (const auto bid : boundary_indicators)
        {
          // gather local arrays into linear vectors for reduction
          std::vector<double> local_total_min = local_min_stress[bid];
          std::vector<double> local_total_max = local_max_stress[bid];
          std::vector<double> local_total_int = local_stress_integral[bid];
          std::vector<double> local_min = local_min_shear_stress[bid];
          std::vector<double> local_max = local_max_shear_stress[bid];
          std::vector<double> local_int = local_shear_stress_integral[bid];
          double local_area = local_boundary_area[bid];

          std::vector<double> g_total_min(local_total_min.size());
          std::vector<double> g_total_max(local_total_max.size());
          std::vector<double> g_total_int(local_total_int.size());
          std::vector<double> gmin(local_min.size());
          std::vector<double> gmax(local_max.size());
          std::vector<double> gint(local_int.size());
          double garea;

          Utilities::MPI::min(local_total_min, this->get_mpi_communicator(), g_total_min);
          Utilities::MPI::max(local_total_max, this->get_mpi_communicator(), g_total_max);
          Utilities::MPI::sum(local_total_int, this->get_mpi_communicator(), g_total_int);

          Utilities::MPI::min(local_min, this->get_mpi_communicator(), gmin);
          Utilities::MPI::max(local_max, this->get_mpi_communicator(), gmax);
          Utilities::MPI::sum(local_int, this->get_mpi_communicator(), gint);
          garea = Utilities::MPI::sum(local_area, this->get_mpi_communicator());

          global_min_stress[bid] = g_total_min;
          global_max_stress[bid] = g_total_max;
          global_avg_stress[bid] = std::vector<double>(g_total_int.size());
          for (unsigned int i=0; i<g_total_int.size(); ++i)
            global_avg_stress[bid][i] = (garea > 0.0 ? g_total_int[i] / garea : 0.0);

          global_min_shear_stress[bid] = gmin;
          global_max_shear_stress[bid] = gmax;
          global_avg_shear_stress[bid] = std::vector<double>(gint.size());
          for (unsigned int i=0; i<gint.size(); ++i)
            global_avg_shear_stress[bid][i] = (garea > 0.0 ? gint[i] / garea : 0.0);
        }

      if constexpr (dim == 3)
        {
          std::vector<std::vector<double>> global_surface_total_stress_sh_cos(n_components,
                                                                               std::vector<double>(n_stress_sh_coefficients, 0.0));
          std::vector<std::vector<double>> global_surface_total_stress_sh_sin(n_components,
                                                                               std::vector<double>(n_stress_sh_coefficients, 0.0));
          std::vector<std::vector<double>> global_surface_deviatoric_stress_sh_cos(n_components,
                                                                                    std::vector<double>(n_stress_sh_coefficients, 0.0));
          std::vector<std::vector<double>> global_surface_deviatoric_stress_sh_sin(n_components,
                                                                                    std::vector<double>(n_stress_sh_coefficients, 0.0));

          for (unsigned int i=0; i<n_components; ++i)
            {
              Utilities::MPI::sum(local_surface_total_stress_sh_cos[i],
                                  this->get_mpi_communicator(),
                                  global_surface_total_stress_sh_cos[i]);
              Utilities::MPI::sum(local_surface_total_stress_sh_sin[i],
                                  this->get_mpi_communicator(),
                                  global_surface_total_stress_sh_sin[i]);
              Utilities::MPI::sum(local_surface_deviatoric_stress_sh_cos[i],
                                  this->get_mpi_communicator(),
                                  global_surface_deviatoric_stress_sh_cos[i]);
              Utilities::MPI::sum(local_surface_deviatoric_stress_sh_sin[i],
                                  this->get_mpi_communicator(),
                                  global_surface_deviatoric_stress_sh_sin[i]);
            }

          std::vector<double> global_surface_tangential_deviatoric_stress_sh_cos(n_stress_sh_coefficients, 0.0);
          std::vector<double> global_surface_tangential_deviatoric_stress_sh_sin(n_stress_sh_coefficients, 0.0);
          Utilities::MPI::sum(local_surface_tangential_deviatoric_stress_sh_cos,
                              this->get_mpi_communicator(),
                              global_surface_tangential_deviatoric_stress_sh_cos);
          Utilities::MPI::sum(local_surface_tangential_deviatoric_stress_sh_sin,
                              this->get_mpi_communicator(),
                              global_surface_tangential_deviatoric_stress_sh_sin);

          Utilities::create_directory(this->get_output_directory() + "surface_stress/",
                                      this->get_mpi_communicator(),
                                      true);

          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              const std::vector<std::string> component_names = stress_component_names_3d;

              const auto write_sh_coefficients =
                [this, n_stress_sh_coefficients]
                (const std::string &filename,
                 const std::string &field_description,
                 const std::vector<double> &cos_coefficients,
                 const std::vector<double> &sin_coefficients)
              {
                Assert(cos_coefficients.size() == n_stress_sh_coefficients,
                       ExcInternalError());
                Assert(sin_coefficients.size() == n_stress_sh_coefficients,
                       ExcInternalError());

                std::ofstream output(this->get_output_directory() + "surface_stress/" + filename);
                output << "# degree order cosine_coefficient sine_coefficient\n";
                output << "# field: " << field_description << ", Pa\n";
                output << "# spherical harmonic normalization: ASPECT real_spherical_harmonic\n";

                for (unsigned int degree = 0, coefficient_index = 0;
                     degree <= stress_sh_max_degree;
                     ++degree)
                  for (unsigned int order = 0; order <= degree; ++order, ++coefficient_index)
                    output << degree << ' '
                           << order << ' '
                           << cos_coefficients[coefficient_index] << ' '
                           << sin_coefficients[coefficient_index] << '\n';
              };

              const std::string timestep_suffix =
                "." + Utilities::int_to_string(this->get_timestep_number(), 5);

              write_sh_coefficients("surface_tangential_deviatoric_stress_SH_coefficients" + timestep_suffix,
                                    "0.5*(deviatoric_stress_tt + deviatoric_stress_pp), geoscience sign convention",
                                    global_surface_tangential_deviatoric_stress_sh_cos,
                                    global_surface_tangential_deviatoric_stress_sh_sin);

              for (unsigned int i=0; i<n_components; ++i)
                {
                  write_sh_coefficients("surface_total_stress_" + component_names[i] + "_SH_coefficients" + timestep_suffix,
                                        "total_stress_" + component_names[i] + ", geoscience sign convention",
                                        global_surface_total_stress_sh_cos[i],
                                        global_surface_total_stress_sh_sin[i]);
                  write_sh_coefficients("surface_deviatoric_stress_" + component_names[i] + "_SH_coefficients" + timestep_suffix,
                                        "deviatoric_stress_" + component_names[i] + ", geoscience sign convention",
                                        global_surface_deviatoric_stress_sh_cos[i],
                                        global_surface_deviatoric_stress_sh_sin[i]);
                }
            }
        }

      const std::vector<std::string> component_names = (dim==2) ? stress_component_names_2d : stress_component_names_3d;

      // Fill statistics table
      for (const auto bid : boundary_indicators)
        {
          const std::string boundary_name = Utilities::int_to_string(bid) +
                                            aspect::Utilities::parenthesize_if_nonempty(this->get_geometry_model().translate_id_to_symbol_name(bid));

          for (unsigned int i=0; i<n_components; ++i)
            {
              const std::string total_base = "Surface stress " + component_names[i] + " on boundary " + boundary_name + " (Pa)";
              const std::string shear_base = "Surface shear stress " + component_names[i] + " on boundary " + boundary_name + " (Pa)";

              const std::array<std::tuple<std::string, double>, 6> table_entries =
              {
                {
                  {"Minimal " + total_base, global_min_stress[bid][i]},
                  {"Average " + total_base, global_avg_stress[bid][i]},
                  {"Maximal " + total_base, global_max_stress[bid][i]},
                  {"Minimal " + shear_base, global_min_shear_stress[bid][i]},
                  {"Average " + shear_base, global_avg_shear_stress[bid][i]},
                  {"Maximal " + shear_base, global_max_shear_stress[bid][i]}
                }
              };

              for (const auto &entry : table_entries)
                {
                  const std::string &name = std::get<0>(entry);
                  statistics.add_value(name, std::get<1>(entry));
                  statistics.set_precision(name, 8);
                  statistics.set_scientific(name, true);
                }
            }
        }

      // Screen text: formatted like StressStatistics (per component, aligned)
      std::ostringstream screen_text;
      screen_text.precision(4);
      screen_text << std::scientific;
      screen_text << "Surface stress min/avg/max (Pa):\n";

      for (const auto bid : boundary_indicators)
        {
          const std::string bname = Utilities::int_to_string(bid) +
                                    aspect::Utilities::parenthesize_if_nonempty(this->get_geometry_model().translate_id_to_symbol_name(bid));

          screen_text << "[" << bname << "]\n";
          for (unsigned int i=0; i<n_components; ++i)
            {
              screen_text << std::setw(2) << std::left << component_names[i] << " total: "
                          << std::right << std::setw(13) << global_min_stress[bid][i] << ' '
                          << std::setw(13) << global_avg_stress[bid][i] << ' '
                          << std::setw(13) << global_max_stress[bid][i] << "\n";
              screen_text << std::setw(2) << std::left << component_names[i] << " dev:   "
                          << std::right << std::setw(13) << global_min_shear_stress[bid][i] << ' '
                          << std::setw(13) << global_avg_shear_stress[bid][i] << ' '
                          << std::setw(13) << global_max_shear_stress[bid][i] << "\n";
            }
        }

      return std::pair<std::string,std::string>("Surface stress statistics:", screen_text.str());
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(SurfaceStressStatistics,
                                  "surface stress statistics",
                                  "A postprocessor that computes min/avg/max statistics of the stress tensor "
                                  "on boundary faces (i.e., surface). Averages are area-weighted. If elasticity "
                                  "is enabled, the deviatoric stress from the material model is used; otherwise "
                                  "it is computed from viscosity and strain rate. If requested via the visualization "
                                  "manager, the tensor is converted to spherical coordinates before statistics are computed.")
  }
}
