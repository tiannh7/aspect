/*
  Copyright (C) 2011 - 2022 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/named_additional_outputs.h>
#include <aspect/material_model/interface.h>
#include <aspect/utilities.h>

#include <algorithm>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      NamedAdditionalOutputs<dim>::
      NamedAdditionalOutputs ()
        :
        DataPostprocessor<dim> (),
        Interface<dim>("")  // physical units depend on run-time parameters
      {}



      template <int dim>
      void
      NamedAdditionalOutputs<dim>::initialize ()
      {
        MaterialModel::MaterialModelOutputs<dim> out(0, this->n_compositional_fields());
        this->get_material_model().create_additional_named_outputs(out);

        AssertThrow(out.additional_outputs.size() > 0,
                    ExcMessage("You activated the postprocessor <named additional outputs>, but there are no additional outputs "
                               "provided by the material model."));

        property_names.clear();
        named_output_indices.clear();

        for (unsigned int k = 0; k < out.additional_outputs.size(); ++k)
          {
            const auto *named_output =
              dynamic_cast<const MaterialModel::NamedAdditionalMaterialOutputs<dim> *>(out.additional_outputs[k].get());

            if (named_output)
              {
                const std::vector<std::string> names = named_output->get_names();
                for (unsigned int i = 0; i < names.size(); ++i)
                  {
                    std::string sanitized_name = names[i];
                    std::replace(sanitized_name.begin(), sanitized_name.end(), ' ', '_');

                    if (selected_property_names.empty() ||
                        std::find(selected_property_names.begin(), selected_property_names.end(), sanitized_name)
                        != selected_property_names.end())
                      {
                        property_names.push_back(sanitized_name);
                        named_output_indices.emplace_back(k, i);
                      }
                  }
              }
          }

        AssertThrow(property_names.size() > 0,
                    ExcMessage("You selected the named additional output postprocessor, but none of the selected "
                               "outputs matched what is provided by the material model."));
      }



      template <int dim>
      std::vector<std::string>
      NamedAdditionalOutputs<dim>::
      get_names () const
      {
        return property_names;
      }



      template <int dim>
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      NamedAdditionalOutputs<dim>::
      get_data_component_interpretation () const
      {
        return std::vector<DataComponentInterpretation::DataComponentInterpretation> (get_names().size(),
                                                                                      DataComponentInterpretation::component_is_scalar);
      }



      template <int dim>
      UpdateFlags
      NamedAdditionalOutputs<dim>::
      get_needed_update_flags () const
      {
        return update_gradients | update_values  | update_quadrature_points;
      }



      template <int dim>
      void
      NamedAdditionalOutputs<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const unsigned int n_q_points = input_data.solution_values.size();
        Assert(computed_quantities.size() == n_q_points, ExcInternalError());

        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_q_points,
                                                     this->n_compositional_fields());

        in.requested_properties = MaterialModel::MaterialProperties::additional_outputs;

        this->get_material_model().create_additional_named_outputs(out);
        this->get_material_model().evaluate(in, out);

        for (unsigned int idx = 0; idx < named_output_indices.size(); ++idx)
          {
            const auto &[k, i] = named_output_indices[idx];

            const auto *named_output =
              dynamic_cast<const MaterialModel::NamedAdditionalMaterialOutputs<dim> *>(out.additional_outputs[k].get());

            Assert(named_output != nullptr, ExcInternalError());

            const std::vector<double> values = named_output->get_nth_output(i);
            for (unsigned int q = 0; q < n_q_points; ++q)
              computed_quantities[q][idx] = values[q];
          }
      }

      template <int dim>
      void
      NamedAdditionalOutputs<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("Named additional outputs");
            {
              const std::string pattern_of_names =
                "current cohesions|"
                "current friction angles|"
                "current yield stresses|"
                "plastic yielding|"
                "diffusion viscosity|"
                "dislocation viscosity|"
                "elastic shear modulus";

              prm.declare_entry("List of named outputs", "",
                                Patterns::MultipleSelection(pattern_of_names),
                                "A comma-separated list of named additional outputs to be visualized. "
                                "If left empty, all available named outputs will be output.\n\n"
                                "The following named outputs are available:\n\n" +
                                pattern_of_names);
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }

      template <int dim>
      void
      NamedAdditionalOutputs<dim>::parse_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("Named additional outputs");
            {
              const std::string param_string = prm.get("List of named outputs");
              if (!param_string.empty())
                {
                  selected_property_names = Utilities::split_string_list(param_string);
                  for (std::string &name : selected_property_names)
                    std::replace(name.begin(), name.end(), ' ', '_');
                }
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }

    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(NamedAdditionalOutputs,
                                                  "named additional outputs",
                                                  "Some material models can compute quantities other than those "
                                                  "that typically appear in the equations that \\aspect{} solves "
                                                  "(such as the viscosity, density, etc). Examples of quantities "
                                                  "material models may be able to compute are seismic velocities, "
                                                  "or other quantities that can be derived from the state variables "
                                                  "and the material coefficients such as the stress or stress "
                                                  "anisotropies. These quantities are generically referred to as "
                                                  "`named outputs' because they are given an explicit name different "
                                                  "from the usual outputs of material models.\n\n"
                                                  "This visualization postprocessor outputs whatever quantities the "
                                                  "material model can compute. What quantities these are is specific "
                                                  "to the material model in use for a simulation, and for many models "
                                                  "in fact does not contain any named outputs at all."
                                                  "\n\n"
                                                  "Physical units: Various, depending on what is being output.")
    }
  }
}
