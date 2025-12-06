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
      NamedAdditionalOutputs<dim>::
      initialize ()
      {
        MaterialModel::MaterialModelOutputs<dim> out(0,
                                                     this->n_compositional_fields());
        this->get_material_model().create_additional_named_outputs(out);

        AssertThrow(out.additional_outputs.size() > 0,
                    ExcMessage("You activated the postprocessor <named additional outputs>, but there are no additional outputs "
                               "provided by the material model. Either remove the postprocessor, or check why no output is provided."));

        std::vector<std::string> available_property_names;
        for (unsigned int k=0; k<out.additional_outputs.size(); ++k)
          {
            const MaterialModel::NamedAdditionalMaterialOutputs<dim> *result
              = dynamic_cast<const MaterialModel::NamedAdditionalMaterialOutputs<dim> *> (out.additional_outputs[k].get());

            if (result)
              {
                std::vector<std::string> names = result->get_names();

                for (const auto &name : names)
                  available_property_names.push_back(name);
              }
          }

        AssertThrow(available_property_names.size() > 0,
                    ExcMessage("You activated the postprocessor <named additional outputs>, but none of the additional outputs "
                               "provided by the material model are named outputs. Either remove the postprocessor, or check why no "
                               "named output is provided."));

        // If no specific outputs are requested, use all available outputs
        if (requested_output_names.empty())
          {
            property_names = available_property_names;
          }
        else
          {
            // Process requested output names (replace spaces with underscores)
            std::vector<std::string> processed_requested_names = requested_output_names;
            for (auto &name : processed_requested_names)
              std::replace(name.begin(), name.end(), ' ', '_');

            // Check that all requested outputs are available
            for (const auto &requested_name : processed_requested_names)
              {
                bool found = false;
                for (const auto &available_name : available_property_names)
                  {
                    if (requested_name == available_name)
                      {
                        found = true;
                        break;
                      }
                  }
                AssertThrow(found,
                            ExcMessage("The requested output variable '" + requested_name + "' is not available. "
                                       "Available outputs are: " + [&]()
                {
                  std::string result;
                  for (size_t i = 0; i < available_property_names.size(); ++i)
                    {
                      if (i > 0)
                        result += ", ";
                      result += available_property_names[i];
                    }
                  return result;
                }()));
              }
            property_names = processed_requested_names;
          }
        for (auto &property_name : property_names)
          std::replace(property_name.begin(),property_name.end(),' ', '_');
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
      NamedAdditionalOutputs<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("Named additional outputs");
            const std::string pattern_of_names = "current cohesions|current friction angles|current yield stresses|plastic yielding|"
                                                 "elastic shear modulus|elastic viscosity|prescribed shear heating rates|"
                                                 "current diffusion viscosity|current dislocation viscosity|"
                                                 "reaction rate C0|reaction rate C1|reaction rate Cn";
            {
              prm.declare_entry("List of named outputs",
                                "",
                                Patterns::MultipleSelection(pattern_of_names),
                                "A comma separated list of named additional outputs that should be "
                                "written whenever writing graphical output. By default, all available "
                                "named additional outputs will be written. If this parameter is set, "
                                "only the specified outputs will be written. The available outputs "
                                "depend on the material model in use. Alternatively, the text `all' "
                                "indicates that all available named additional outputs should be written.");
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
              requested_output_names = Utilities::split_string_list(prm.get("List of named outputs"));
              // Convert spaces to underscores to match the internal naming convention
              for (auto &name : requested_output_names)
                std::replace(name.begin(), name.end(), ' ', '_');
              AssertThrow(Utilities::has_unique_entries(requested_output_names),
                          ExcMessage("The list of strings for the parameter "
                                     "'Postprocess/Visualization/Named additional outputs/List of named outputs' "
                                     "contains entries more than once. This is not allowed. "
                                     "Please check your parameter file."));
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

        // see if 'all' was selected (or is part of the list). if so
        // simply replace the list with one that contains all names
        if (std::find (requested_output_names.begin(),
                       requested_output_names.end(),
                       "all") != requested_output_names.end())
          {
            requested_output_names.clear();
            // We will handle this in initialize() by using all available outputs
          }
      }



      template <int dim>
      void
      NamedAdditionalOutputs<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,
                ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,
                ExcInternalError());

        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        in.requested_properties = MaterialModel::MaterialProperties::additional_outputs;

        this->get_material_model().create_additional_named_outputs(out);
        this->get_material_model().evaluate(in, out);

        unsigned int field_index = 0;
        for (unsigned int k=0; k<out.additional_outputs.size(); ++k)
          {
            const MaterialModel::NamedAdditionalMaterialOutputs<dim> *result
              = dynamic_cast<const MaterialModel::NamedAdditionalMaterialOutputs<dim> *> (out.additional_outputs[k].get());

            if (result)
              {
                std::vector<std::string> output_names = result->get_names();
                for (unsigned int i=0; i<output_names.size(); ++i)
                  {
                    // Process the output name (replace spaces with underscores)
                    std::string processed_output_name = output_names[i];
                    std::replace(processed_output_name.begin(), processed_output_name.end(), ' ', '_');

                    // Check if this output is requested
                    bool is_requested = false;
                    if (requested_output_names.empty())
                      {
                        // If no specific outputs requested, output all
                        is_requested = true;
                      }
                    else
                      {
                        // Check if the processed name is in property_names
                        is_requested = std::find(property_names.begin(), property_names.end(), processed_output_name) != property_names.end();
                      }

                    if (is_requested)
                      {
                        std::vector<double> outputs = result->get_nth_output(i);

                        for (unsigned int q=0; q<n_quadrature_points; ++q)
                          computed_quantities[q][field_index] = outputs[q];

                        ++field_index;
                      }
                  }
              }
          }
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
