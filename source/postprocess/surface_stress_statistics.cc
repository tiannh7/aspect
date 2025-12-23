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

#include <map>
#include <iomanip>


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

      // Maps keyed by boundary id to arrays over tensor components (shear stress)
      std::map<types::boundary_id, std::vector<double>> local_min_shear_stress;
      std::map<types::boundary_id, std::vector<double>> local_max_shear_stress;
      std::map<types::boundary_id, std::vector<double>> local_shear_stress_integral;
      std::map<types::boundary_id, double> local_boundary_area;

      const unsigned int n_components = SymmetricTensor<2,dim>::n_independent_components;
      const std::set<types::boundary_id> boundary_indicators = this->get_geometry_model().get_used_boundary_indicators();
      for (const auto id : boundary_indicators)
        {
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

                    // Shear stress is the deviatoric part, with geoscience sign convention
                    SymmetricTensor<2,dim> shear_stress = -deviatoric_stress;
                    if (output_spherical || is_spherical_like)
                      shear_stress = - Utilities::Coordinates::cartesian_to_spherical_tensor(deviatoric_stress, in.position[q]);

                    for (unsigned int i=0; i<n_components; ++i)
                      {
                        const TableIndices<2> idx = SymmetricTensor<2,dim>::unrolled_to_component_indices(i);
                        const double val = shear_stress[idx[0]][idx[1]];
                        local_min_shear_stress[bid][i] = std::min(local_min_shear_stress[bid][i], val);
                        local_max_shear_stress[bid][i] = std::max(local_max_shear_stress[bid][i], val);
                        local_shear_stress_integral[bid][i] += val * fe_face_values.JxW(q);
                      }
                    local_boundary_area[bid] += fe_face_values.JxW(q);
                  }
              }

      // MPI reductions per boundary id
      std::map<types::boundary_id, std::vector<double>> global_min_shear_stress;
      std::map<types::boundary_id, std::vector<double>> global_max_shear_stress;
      std::map<types::boundary_id, std::vector<double>> global_avg_shear_stress;

      for (const auto bid : boundary_indicators)
        {
          // gather local arrays into linear vectors for reduction
          std::vector<double> local_min = local_min_shear_stress[bid];
          std::vector<double> local_max = local_max_shear_stress[bid];
          std::vector<double> local_int = local_shear_stress_integral[bid];
          double local_area = local_boundary_area[bid];

          std::vector<double> gmin(local_min.size());
          std::vector<double> gmax(local_max.size());
          std::vector<double> gint(local_int.size());
          double garea;

          Utilities::MPI::min(local_min, this->get_mpi_communicator(), gmin);
          Utilities::MPI::max(local_max, this->get_mpi_communicator(), gmax);
          Utilities::MPI::sum(local_int, this->get_mpi_communicator(), gint);
          garea = Utilities::MPI::sum(local_area, this->get_mpi_communicator());

          global_min_shear_stress[bid] = gmin;
          global_max_shear_stress[bid] = gmax;
          global_avg_shear_stress[bid] = std::vector<double>(gint.size());
          for (unsigned int i=0; i<gint.size(); ++i)
            global_avg_shear_stress[bid][i] = (garea > 0.0 ? gint[i] / garea : 0.0);
        }

      const std::vector<std::string> component_names = (dim==2) ? stress_component_names_2d : stress_component_names_3d;

      // Fill statistics table
      for (const auto bid : boundary_indicators)
        {
          const std::string boundary_name = Utilities::int_to_string(bid) +
                                            aspect::Utilities::parenthesize_if_nonempty(this->get_geometry_model().translate_id_to_symbol_name(bid));

          for (unsigned int i=0; i<n_components; ++i)
            {
              const std::string base = "Surface shear stress " + component_names[i] + " on boundary " + boundary_name + " (Pa)";

              const std::string name_min = "Minimal " + base;
              const std::string name_avg = "Average " + base;
              const std::string name_max = "Maximal " + base;

              statistics.add_value(name_min, global_min_shear_stress[bid][i]);
              statistics.add_value(name_avg, global_avg_shear_stress[bid][i]);
              statistics.add_value(name_max, global_max_shear_stress[bid][i]);

              statistics.set_precision(name_min, 8);
              statistics.set_scientific(name_min, true);
              statistics.set_precision(name_avg, 8);
              statistics.set_scientific(name_avg, true);
              statistics.set_precision(name_max, 8);
              statistics.set_scientific(name_max, true);
            }
        }

      // Screen text: formatted like StressStatistics (per component, aligned)
      std::ostringstream screen_text;
      screen_text.precision(4);
      screen_text << std::scientific;
      screen_text << "Surface shear stress min/avg/max (Pa):\n";

      for (const auto bid : boundary_indicators)
        {
          const std::string bname = Utilities::int_to_string(bid) +
                                    aspect::Utilities::parenthesize_if_nonempty(this->get_geometry_model().translate_id_to_symbol_name(bid));

          screen_text << "[" << bname << "]\n";
          for (unsigned int i=0; i<n_components; ++i)
            {
              screen_text << std::setw(2) << std::left << component_names[i] << ": "
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
