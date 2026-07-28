/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/material_model/viscoelastic.h>
#include <aspect/utilities.h>
#include <aspect/global.h>

#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_q1.h>

#include <algorithm>
#include <numeric>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    ViscoelasticAsciiProfileReferencePositions<dim>::
    ViscoelasticAsciiProfileReferencePositions(
      const unsigned int n_points,
      const Mapping<dim> &current_mapping,
      const Mapping<dim> &reference_mapping)
      :
      positions(n_points),
      current_mapping(&current_mapping),
      reference_mapping(&reference_mapping)
    {}



    template <int dim>
    void
    ViscoelasticAsciiProfileReferencePositions<dim>::
    fill(const LinearAlgebra::BlockVector &,
         const FEValuesBase<dim> &fe_values,
         const Introspection<dim> &)
    {
      positions.resize(fe_values.n_quadrature_points);

      const FEValues<dim> *cell_fe_values =
        dynamic_cast<const FEValues<dim> *>(&fe_values);
      if (cell_fe_values != nullptr)
        {
          if (reference_fe_values == nullptr)
            reference_fe_values = std::make_unique<FEValues<dim>>(
                                    *reference_mapping,
                                    fe_values.get_fe(),
                                    cell_fe_values->get_quadrature(),
                                    update_quadrature_points);

          reference_fe_values->reinit(fe_values.get_cell());
          positions = reference_fe_values->get_quadrature_points();
          return;
        }

      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          const Point<dim> unit_position =
            current_mapping->transform_real_to_unit_cell(fe_values.get_cell(),
                                                         fe_values.quadrature_point(q));
          positions[q] =
            reference_mapping->transform_unit_to_real_cell(fe_values.get_cell(),
                                                           unit_position);
        }
    }



    template <int dim>
    Viscoelastic<dim>::Viscoelastic ()
      :
      use_ascii_profile(false),
      use_reference_geometry_for_ascii_profile(false),
      use_reference_cell_center_for_ascii_profile(false),
      profile_density_index(numbers::invalid_unsigned_int),
      profile_viscosity_index(numbers::invalid_unsigned_int),
      profile_elastic_shear_modulus_index(numbers::invalid_unsigned_int),
      profile_compressibility_index(numbers::invalid_unsigned_int),
      profile_elastic_lame_lambda_index(numbers::invalid_unsigned_int),
      radial_displacement_field_index(numbers::invalid_unsigned_int)
    {}



    template <int dim>
    void
    Viscoelastic<dim>::initialize ()
    {
      if (use_ascii_profile
          && use_reference_geometry_for_ascii_profile
          && this->get_parameters().mesh_deformation_enabled)
        {
          if (this->get_geometry_model().has_curved_elements())
            ascii_profile_reference_mapping = std::make_unique<MappingQGeneric<dim>>(4);
          else
            ascii_profile_reference_mapping = std::make_unique<MappingQ1<dim>>();
        }

      if (use_ascii_profile)
        {
          material_profile.initialize(this->get_mpi_communicator());
          profile_density_index = material_profile.get_column_index_from_name("density");
          profile_viscosity_index = material_profile.get_column_index_from_name("viscosity");
          profile_elastic_shear_modulus_index =
            material_profile.get_column_index_from_name("elastic_shear_modulus");
          profile_compressibility_index =
            material_profile.maybe_get_column_index_from_name("compressibility");
          profile_elastic_lame_lambda_index =
            material_profile.maybe_get_column_index_from_name("lame_lambda");

          if (enable_compressible_maxwell)
            {
              AssertThrow(profile_compressibility_index != numbers::invalid_unsigned_int
                          || profile_elastic_lame_lambda_index != numbers::invalid_unsigned_int,
                          ExcMessage("A viscoelastic ASCII profile with compressible Maxwell elasticity "
                                     "requires a 'compressibility' or 'lame_lambda' column."));

              for (const double depth : material_profile.get_interpolation_point_coordinates())
                {
                  const Point<1> profile_position(depth);
                  const double shear_modulus =
                    material_profile.get_data_component(profile_position,
                                                        profile_elastic_shear_modulus_index);
                  double bulk_modulus_from_compressibility = numbers::signaling_nan<double>();
                  double bulk_modulus_from_lame_lambda = numbers::signaling_nan<double>();

                  if (profile_compressibility_index != numbers::invalid_unsigned_int)
                    {
                      const double compressibility =
                        material_profile.get_data_component(profile_position,
                                                            profile_compressibility_index);
                      AssertThrow(std::isfinite(compressibility) && compressibility > 0.0,
                                  ExcMessage("The viscoelastic ASCII profile requires positive finite compressibility values."));
                      bulk_modulus_from_compressibility = 1.0 / compressibility;
                    }

                  if (profile_elastic_lame_lambda_index != numbers::invalid_unsigned_int)
                    {
                      const double lame_lambda =
                        material_profile.get_data_component(profile_position,
                                                            profile_elastic_lame_lambda_index);
                      bulk_modulus_from_lame_lambda = lame_lambda + 2.0 * shear_modulus / 3.0;
                      AssertThrow(std::isfinite(bulk_modulus_from_lame_lambda)
                                  && bulk_modulus_from_lame_lambda > 0.0,
                                  ExcMessage("The viscoelastic ASCII profile requires lame_lambda+2G/3 to be positive and finite."));
                    }

                  if (profile_compressibility_index != numbers::invalid_unsigned_int
                      && profile_elastic_lame_lambda_index != numbers::invalid_unsigned_int)
                    {
                      const double relative_difference =
                        std::abs(bulk_modulus_from_compressibility - bulk_modulus_from_lame_lambda)
                        / std::max(bulk_modulus_from_compressibility,
                                   bulk_modulus_from_lame_lambda);
                      AssertThrow(relative_difference <= 1e-6,
                                  ExcMessage("The viscoelastic ASCII profile columns 'compressibility' and "
                                             "'lame_lambda' are inconsistent with K=lambda+2G/3."));
                    }
                }
            }
        }
    }



    template <int dim>
    void
    Viscoelastic<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      EquationOfStateOutputs<dim> eos_outputs (this->introspection().get_number_of_fields_of_type(CompositionalFieldDescription::chemical_composition)+1);

      const std::shared_ptr<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>
                                                                                 additional_stokes_rhs =
                                                                                   out.template get_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>();

      std::vector<double> average_elastic_shear_moduli (in.n_evaluation_points());
      std::vector<double> elastic_shear_moduli(elastic_rheology.get_elastic_shear_moduli());
      std::vector<double> average_elastic_bulk_moduli(in.n_evaluation_points(),
                                                      numbers::signaling_nan<double>());
      const std::shared_ptr<const ViscoelasticAsciiProfileReferencePositions<dim>>
                                                                                reference_positions =
                                                                                  in.template get_additional_input_object<
                                                                                  ViscoelasticAsciiProfileReferencePositions<dim>>();

      Point<1> reference_cell_center_profile_position;
      const bool sample_ascii_profile_at_reference_cell_center =
        use_ascii_profile
        && use_reference_cell_center_for_ascii_profile
        && this->get_parameters().mesh_deformation_enabled
        && in.current_cell.state() == IteratorState::valid;

      if (sample_ascii_profile_at_reference_cell_center)
        {
          Assert(ascii_profile_reference_mapping != nullptr,
                 ExcInternalError());
          Point<dim> unit_cell_center;
          for (unsigned int d = 0; d < dim; ++d)
            unit_cell_center[d] = 0.5;
          const Point<dim> reference_cell_center =
            ascii_profile_reference_mapping->transform_unit_to_real_cell(in.current_cell,
                                                                         unit_cell_center);
          reference_cell_center_profile_position[0] =
            this->get_geometry_model().depth(reference_cell_center);
        }

      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          const std::vector<double> composition = in.composition[i];

          const std::vector<double> volume_fractions = MaterialUtilities::compute_only_composition_fractions(composition,
                                                       this->introspection().chemical_composition_field_indices());

          equation_of_state.evaluate(in, i, eos_outputs);

          // Arithmetic averaging of thermal conductivities
          // This may not be strictly the most reasonable thing, but for most Earth materials we hope
          // that they do not vary so much that it is a big problem.
          out.thermal_conductivities[i] = MaterialUtilities::average_value(volume_fractions, thermal_conductivities, MaterialUtilities::arithmetic);

          // not strictly correct if thermal expansivities are different, since we are interpreting
          // these compositions as volume fractions, but the error introduced should not be too bad.
          out.densities[i] = MaterialUtilities::average_value (volume_fractions, eos_outputs.densities, MaterialUtilities::arithmetic);
          out.thermal_expansion_coefficients[i] = MaterialUtilities::average_value (volume_fractions, eos_outputs.thermal_expansion_coefficients, MaterialUtilities::arithmetic);
          out.specific_heat[i] = MaterialUtilities::average_value (volume_fractions, eos_outputs.specific_heat_capacities, MaterialUtilities::arithmetic);

          out.compressibilities[i] = MaterialUtilities::average_value (volume_fractions, eos_outputs.compressibilities, MaterialUtilities::arithmetic);
          out.entropy_derivative_pressure[i] = MaterialUtilities::average_value (volume_fractions, eos_outputs.entropy_derivative_pressure, MaterialUtilities::arithmetic);
          out.entropy_derivative_temperature[i] = MaterialUtilities::average_value (volume_fractions, eos_outputs.entropy_derivative_temperature, MaterialUtilities::arithmetic);

          Point<1> profile_position;
          if (use_ascii_profile)
            {
              if (sample_ascii_profile_at_reference_cell_center)
                profile_position = reference_cell_center_profile_position;
              else
                {
                  Point<dim> profile_sample_position = in.position[i];
                  if (use_reference_geometry_for_ascii_profile
                      && this->get_parameters().mesh_deformation_enabled
                      && in.current_cell.state() == IteratorState::valid)
                    {
                      if (reference_positions != nullptr)
                        profile_sample_position = reference_positions->positions[i];
                      else
                        {
                          Assert(ascii_profile_reference_mapping != nullptr,
                                 ExcInternalError());
                          const Point<dim> unit_position =
                            this->get_mapping().transform_real_to_unit_cell(in.current_cell,
                                                                            in.position[i]);
                          profile_sample_position =
                            ascii_profile_reference_mapping->transform_unit_to_real_cell(in.current_cell,
                                                                                         unit_position);
                        }
                    }

                  profile_position[0] =
                    this->get_geometry_model().depth(profile_sample_position);
                }
              out.densities[i] =
                material_profile.get_data_component(profile_position, profile_density_index);
            }

          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;

          if (additional_stokes_rhs != nullptr &&
              this->get_parameters().stokes_pressure_formulation_is_dynamic &&
              this->get_parameters().stokes_pressure_reference_density > 0.0)
            {
              if (out.densities[i] < this->get_parameters().stokes_pressure_reference_density * 0.5)
                {
                  AssertThrow(false, ExcMessage("Double subtraction detected: Stokes RHS includes reference density subtraction, "
                                                "but the material model density is already perturbation density."));
                }
              additional_stokes_rhs->rhs_u[i] =
                -this->get_parameters().stokes_pressure_reference_density
                * this->get_gravity_model().gravity_vector(in.position[i]);
            }

          // Average the viscous viscosity and the shear modulus over the compositions
          if (use_ascii_profile)
            average_elastic_shear_moduli[i] =
              material_profile.get_data_component(profile_position,
                                                  profile_elastic_shear_modulus_index);
          else
            average_elastic_shear_moduli[i] =
              MaterialUtilities::average_value(volume_fractions,
                                               elastic_shear_moduli,
                                               viscosity_averaging);

          if (enable_compressible_maxwell)
            {
              if (use_ascii_profile)
                {
                  if (profile_compressibility_index != numbers::invalid_unsigned_int)
                    average_elastic_bulk_moduli[i] =
                      1.0 / material_profile.get_data_component(profile_position,
                                                                profile_compressibility_index);
                  else
                    average_elastic_bulk_moduli[i] =
                      material_profile.get_data_component(profile_position,
                                                          profile_elastic_lame_lambda_index)
                      + 2.0 * average_elastic_shear_moduli[i] / 3.0;
                }
              else
                {
                  std::vector<double> phase_bulk_moduli(elastic_bulk_moduli);
                  if (elastic_bulk_modulus_formulation == ElasticBulkModulusFormulation::lame_lambda)
                    for (unsigned int j = 0; j < phase_bulk_moduli.size(); ++j)
                      phase_bulk_moduli[j] = elastic_lame_lambda_moduli[j]
                                             + 2.0 * elastic_shear_moduli[j] / 3.0;

                  average_elastic_bulk_moduli[i] =
                    MaterialUtilities::average_value(volume_fractions,
                                                     phase_bulk_moduli,
                                                     MaterialUtilities::arithmetic);
                }

              AssertThrow(std::isfinite(average_elastic_bulk_moduli[i])
                          && average_elastic_bulk_moduli[i] > 0.0,
                          ExcMessage("Compressible Maxwell elasticity requires a finite positive elastic bulk modulus."));
              out.compressibilities[i] = 1.0 / average_elastic_bulk_moduli[i];
            }

          // If we have multiple compositions, we need to first compute their respective viscoelastic viscosities,
          // based on their respective viscous viscosities and the averaged shear modulus, before averaging them
          // into the final effective viscosity.
          if (use_ascii_profile)
            {
              const double viscosity =
                material_profile.get_data_component(profile_position,
                                                    profile_viscosity_index);
              out.viscosities[i] =
                elastic_rheology.calculate_viscoelastic_viscosity(viscosity,
                                                                  average_elastic_shear_moduli[i]);
            }
          else
            {
              std::vector<double> viscoelastic_viscosities(volume_fractions.size());
              for (unsigned int j=0; j < volume_fractions.size(); ++j)
                {
                  // The viscoelastic viscosity is scaled with the timestep ratio $\frac{\Delta t_c}{\Delta t_{el}}$ in the
                  // calculate_viscoelastic_viscosity function.
                  viscoelastic_viscosities[j] = elastic_rheology.calculate_viscoelastic_viscosity(viscosities[j],
                                                                                                  average_elastic_shear_moduli[i]);
                }

              // Average viscoelastic (e.g., effective) viscosity (equation 28 in Moresi et al., 2003, J. Comp. Phys.).
              out.viscosities[i] = MaterialUtilities::average_value(volume_fractions,
                                                                    viscoelastic_viscosities,
                                                                    viscosity_averaging);
            }
        }

      // Fill the body force term, viscoelastic strain rate and viscous dissipation.
      elastic_rheology.fill_elastic_outputs(in, average_elastic_shear_moduli, out);

      if (enable_compressible_maxwell)
        {
          const std::shared_ptr<MaterialModel::ElasticOutputs<dim>> elastic_outputs =
                                                                   out.template get_additional_output_object<MaterialModel::ElasticOutputs<dim>>();
          if (elastic_outputs != nullptr)
            elastic_outputs->elastic_bulk_moduli = average_elastic_bulk_moduli;
        }
      // Fill the elastic additional outputs with the shear modulus, elastic viscosity
      // and deviatoric stress of the current timestep.
      // TODO requests_property is already checked in the fill_ function,
      // but we can also do it here
      //if (in.requests_property(MaterialProperties::additional_outputs))
      elastic_rheology.fill_elastic_additional_outputs(in, average_elastic_shear_moduli, out);
      // Fill the reaction terms to apply the rotation of the stresses into the current timestep.
      elastic_rheology.fill_reaction_outputs(in, average_elastic_shear_moduli, out);
      // Fill the reaction_rates that during operator splitting apply the stress update of the previous
      // timestep to the advected and rotated stress computed in the previous timestep ($\tau^{0adv}$)
      // to obtain $\tau^{t}$.
      elastic_rheology.fill_reaction_rates(in, average_elastic_shear_moduli, out);

      const bool use_radial_displacement_history =
        this->get_parameters().density_source_law
        == Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation;
      if (use_radial_displacement_history)
        {
          const std::shared_ptr<ReactionRateOutputs<dim>> reaction_rate_out =
                                                         out.template get_additional_output_object<ReactionRateOutputs<dim>>();

          if (reaction_rate_out != nullptr
              && in.current_cell.state() == IteratorState::valid
              && this->get_timestep_number() > 0
              && !(use_instantaneous_elastic_response_at_timestep_zero()
                   && this->get_timestep_number() == 1)
              && (in.requests_property(MaterialProperties::reaction_rates)
                  || in.requests_property(MaterialProperties::additional_outputs)))
            for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
              {
                const double radius = in.position[i].norm();
                AssertThrow(radius > 0.0,
                            ExcMessage("The radial material-displacement history is undefined at radius zero."));
                const Tensor<1,dim> radial_unit = in.position[i] / radius;
                reaction_rate_out->reaction_rates[i][radial_displacement_field_index] =
                  in.velocity[i] * radial_unit;
              }
        }
    }



    template <int dim>
    bool
    Viscoelastic<dim>::
    is_compressible () const
    {
      return enable_compressible_maxwell || equation_of_state.is_compressible();
    }



    template <int dim>
    void
    Viscoelastic<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Viscoelastic");
        {
          EquationOfState::MulticomponentIncompressible<dim>::declare_parameters (prm);
          Rheology::Elasticity<dim>::declare_parameters (prm);

          prm.declare_entry ("Viscosities", "1.e21",
                             Patterns::List(Patterns::Double (0.)),
                             "List of viscosities for background mantle and compositional fields, "
                             "for a total of N+1 values, where N is the number of all compositional fields or only "
                             "those corresponding to chemical compositions. "
                             "If only one value is given, then all use the same value. "
                             "Units: $\\text{Pa}\\text{s}$.");
          prm.declare_entry ("Thermal conductivities", "4.7",
                             Patterns::List(Patterns::Double (0.)),
                             "List of thermal conductivities for background mantle and compositional fields, "
                             "for a total of N+1 values, where N is the number of all compositional fields or only "
                             "those corresponding to chemical compositions. "
                             "If only one value is given, then all use the same value. "
                             "Units: $\\frac{\\text{W}}{\\text{m}\\text{K}}$.");
          prm.declare_entry ("Use ascii profile", "false",
                             Patterns::Bool(),
                             "Whether to read density, viscosity, and elastic shear modulus "
                             "from a one-dimensional ASCII depth profile. The required column "
                             "names are `density', `viscosity', and `elastic_shear_modulus'. "
                             "When compressible Maxwell elasticity is enabled, the profile must "
                             "also provide `compressibility' or `lame_lambda'; these define the "
                             "elastic bulk modulus without changing the thermodynamic equation "
                             "of state. If both columns are present, they must satisfy "
                             "K=1/compressibility=lame_lambda+2G/3. This option is disabled by "
                             "default and does not by itself enable finite compressibility.");
          prm.declare_entry ("Use reference geometry for ascii profile", "false",
                             Patterns::Bool(),
                             "Whether the depth used to sample the one-dimensional ASCII "
                             "material profile is measured on the undeformed reference mesh "
                             "instead of the current ALE mapping. This option changes only "
                             "the material-profile lookup coordinate: velocity gradients, "
                             "quadrature weights, and all other kinematic quantities continue "
                             "to use the current deformed mesh. It is useful for interface-fitted "
                             "layered models whose material interfaces should remain attached "
                             "to their reference mesh layers while the ALE mesh deforms. The "
                             "default preserves the existing current-geometry behavior.");
          prm.declare_entry ("Use reference cell center for ascii profile", "false",
                             Patterns::Bool(),
                             "Whether all material evaluation points in a cell sample "
                             "the ASCII profile at the center of that cell on the "
                             "undeformed reference mesh. This is intended for elemental "
                             "profiles whose density and rheology are assigned once per "
                             "reference cell, including discontinuous layered profiles. "
                             "It requires 'Use reference geometry for ascii profile = true'. "
                             "The default samples the profile at each evaluation point.");
          aspect::Utilities::AsciiDataProfile<dim>::declare_parameters(prm,
                                                                       "$ASPECT_SOURCE_DIR/data/material-model/viscoelastic/",
                                                                       "viscoelastic_profile.txt",
                                                                       "Ascii profile");
          prm.declare_entry ("Enable compressible Maxwell", "false",
                             Patterns::Bool(),
                             "Whether to enable a finite elastic bulk response. "
                             "When enabled, the material model supplies an elastic "
                             "bulk modulus to the `elastic pressure evolution' mass "
                             "formulation. Physical material density remains the "
                             "incompressible equation-of-state density. This option "
                             "is disabled by default.");
          prm.declare_entry ("Elastic bulk modulus formulation", "bulk modulus",
                             Patterns::Selection("bulk modulus|Lame lambda"),
                             "Select whether elastic bulk modulus K is specified "
                             "directly or computed as K=lambda+2G/3 from Lame's "
                             "first parameter and the elastic shear modulus. This "
                             "selection applies to parameter-list input; ASCII "
                             "profiles use their compressibility or lame_lambda column.");
          prm.declare_entry ("Elastic bulk moduli", "2.e11",
                             Patterns::List(Patterns::Double (0.)),
                             "Elastic bulk moduli for the background and chemical "
                             "compositions. A single value applies to every phase. "
                             "Units: Pa.");
          prm.declare_entry ("Elastic Lame lambda moduli", "1.5e11",
                             Patterns::List(Patterns::Double (0.)),
                             "Lame's first parameters for the background and chemical "
                             "compositions. These values are used when `Elastic bulk "
                             "modulus formulation' is `Lame lambda'. A single value "
                             "applies to every phase. Units: Pa.");
          prm.declare_entry ("Viscosity averaging scheme", "harmonic",
                             Patterns::Selection("arithmetic|harmonic|geometric|maximum composition "),
                             "When more than one compositional field is present at a point "
                             "with different viscosities, we need to come up with an average "
                             "viscosity at that point.  Select a weighted harmonic, arithmetic, "
                             "geometric, or maximum composition.");
          prm.declare_entry ("Reference density for Stokes perturbation", "0",
                             Patterns::Double (0.),
                             "Constant reference density whose gravitational body force is "
                             "subtracted through the additional Stokes RHS, so the Stokes "
                             "equation sees (rho - rho_ref)*g rather than rho*g. "
                             "A value of zero leaves the full-pressure formulation unchanged. "
                             "This permits an incompressible perturbation-pressure weak form "
                             "while retaining physical material density for other model components "
                             "(e.g. geoid, self-gravity). Units: kg/m^3.");
          prm.declare_entry ("Reference density for perturbation Stokes", "-1.e300",
                             Patterns::Double (),
                             "Deprecated. Use 'Reference density for Stokes perturbation' instead.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    Viscoelastic<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Viscoelastic");
        {
          // Equation of state parameters
          equation_of_state.initialize_simulator (this->get_simulator());
          equation_of_state.parse_parameters (prm);

          elastic_rheology.initialize_simulator (this->get_simulator());
          elastic_rheology.parse_parameters(prm);

          viscosity_averaging = MaterialUtilities::parse_compositional_averaging_operation ("Viscosity averaging scheme",
                                prm);

          // Make options file for parsing maps to double arrays
          std::vector<std::string> chemical_field_names = this->introspection().chemical_composition_field_names();
          chemical_field_names.insert(chemical_field_names.begin(),"background");

          std::vector<std::string> compositional_field_names = this->introspection().get_composition_names();
          compositional_field_names.insert(compositional_field_names.begin(),"background");

          Utilities::MapParsing::Options options(chemical_field_names, "Viscosities");
          options.list_of_allowed_keys = compositional_field_names;

          viscosities = Utilities::MapParsing::parse_map_to_double_array (prm.get("Viscosities"), options);
          options.property_name = "Thermal conductivities";
          thermal_conductivities = Utilities::MapParsing::parse_map_to_double_array (prm.get("Thermal conductivities"), options);
          use_ascii_profile = prm.get_bool("Use ascii profile");
          use_reference_geometry_for_ascii_profile =
            prm.get_bool("Use reference geometry for ascii profile");
          use_reference_cell_center_for_ascii_profile =
            prm.get_bool("Use reference cell center for ascii profile");
          material_profile.parse_parameters(prm, "Ascii profile");
          enable_compressible_maxwell = prm.get_bool("Enable compressible Maxwell");

          AssertThrow(!use_reference_geometry_for_ascii_profile
                      || use_ascii_profile,
                      ExcMessage("'Use reference geometry for ascii profile' "
                                 "requires 'Use ascii profile = true'."));
          AssertThrow(!use_reference_cell_center_for_ascii_profile
                      || use_reference_geometry_for_ascii_profile,
                      ExcMessage("'Use reference cell center for ascii profile' "
                                 "requires 'Use reference geometry for ascii profile = true'."));

          const std::string bulk_modulus_formulation =
            prm.get("Elastic bulk modulus formulation");
          if (bulk_modulus_formulation == "bulk modulus")
            elastic_bulk_modulus_formulation = ElasticBulkModulusFormulation::bulk_modulus;
          else if (bulk_modulus_formulation == "Lame lambda")
            elastic_bulk_modulus_formulation = ElasticBulkModulusFormulation::lame_lambda;
          else
            AssertThrow(false, ExcNotImplemented());

          options.property_name = "Elastic bulk moduli";
          elastic_bulk_moduli =
            Utilities::MapParsing::parse_map_to_double_array(prm.get("Elastic bulk moduli"), options);
          options.property_name = "Elastic Lame lambda moduli";
          elastic_lame_lambda_moduli =
            Utilities::MapParsing::parse_map_to_double_array(prm.get("Elastic Lame lambda moduli"), options);

          if (enable_compressible_maxwell && !use_ascii_profile)
            {
              const std::vector<double> &selected_values =
                (elastic_bulk_modulus_formulation == ElasticBulkModulusFormulation::bulk_modulus
                 ? elastic_bulk_moduli
                 : elastic_lame_lambda_moduli);
              for (const double value : selected_values)
                AssertThrow(value > 0.0,
                            ExcMessage("Compressible Maxwell elasticity requires positive bulk or Lame moduli."));
            }

          if (this->get_parameters().density_source_law
              == Parameters<dim>::Formulation::DensitySourceLaw::mechanical_mass_conservation)
            {
              AssertThrow(enable_compressible_maxwell,
                          ExcMessage("Mechanical mass conservation requires <Enable compressible Maxwell = true>."));
              AssertThrow(this->get_parameters().use_operator_splitting,
                          ExcMessage("Mechanical mass conservation requires operator splitting for radial material-displacement history."));
              AssertThrow(this->introspection().compositional_name_exists("ve_radial_displacement"),
                          ExcMessage("Mechanical mass conservation requires a compositional field named `ve_radial_displacement'."));

              radial_displacement_field_index =
                this->introspection().compositional_index_for_name("ve_radial_displacement");
              const auto radial_displacement_method =
                this->get_parameters().compositional_field_methods[radial_displacement_field_index];
              AssertThrow(radial_displacement_method == Parameters<dim>::AdvectionFieldMethod::fem_field
                          || radial_displacement_method == Parameters<dim>::AdvectionFieldMethod::static_field,
                          ExcMessage("The radial material-displacement history must use either the `field' "
                                     "or `static' compositional field method. Use `static' when the history "
                                     "is defined on the reference mesh and must be updated by reactions "
                                     "without compositional advection."));
              AssertThrow(this->get_parameters().use_discontinuous_composition_discretization[radial_displacement_field_index],
                          ExcMessage("The radial material-displacement history requires a discontinuous compositional element."));
              AssertThrow(this->get_parameters().composition_descriptions[radial_displacement_field_index].type
                          == CompositionalFieldDescription::generic,
                          ExcMessage("The radial material-displacement history must use the `generic' compositional field type."));
            }
          const double new_ref = prm.get_double("Reference density for Stokes perturbation");
          const double old_ref = prm.get_double("Reference density for perturbation Stokes");
          if (old_ref != -1e300 || new_ref != 0.0)
            {
              dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
              pcout << "WARNING: Parameter <Material model/Viscoelastic/Reference density for Stokes perturbation> "
                    << "and its deprecated alias <Reference density for perturbation Stokes> are no longer used. "
                    << "Please use <Formulation/Stokes pressure/Pressure formulation = dynamic pressure> "
                    << "and <Formulation/Stokes pressure/Reference density> instead. "
                    << "The material-model parameter is now ignored." << std::endl;
            }
          // Note: reference_density_for_stokes_perturbation is kept as a member to avoid
          // removing it from the header in one step, but the body-force logic in evaluate()
          // now reads from this->get_parameters().stokes_pressure_reference_density.
          reference_density_for_stokes_perturbation = 0.0;
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();



      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::compositional_fields;
      this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::compositional_fields;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::compositional_fields;
      this->model_dependence.thermal_conductivity = NonlinearDependence::compositional_fields;
    }



    template <int dim>
    void
    Viscoelastic<dim>::fill_additional_material_model_inputs(
      MaterialModel::MaterialModelInputs<dim> &input,
      const LinearAlgebra::BlockVector &solution,
      const FEValuesBase<dim> &fe_values,
      const Introspection<dim> &introspection) const
    {
      if (use_ascii_profile
          && use_reference_geometry_for_ascii_profile
          && this->get_parameters().mesh_deformation_enabled
          && !input.template has_additional_input_object<
          ViscoelasticAsciiProfileReferencePositions<dim>>())
        {
          Assert(ascii_profile_reference_mapping != nullptr,
                 ExcInternalError());
          input.additional_inputs.emplace_back(
            std::make_unique<ViscoelasticAsciiProfileReferencePositions<dim>>(
              input.n_evaluation_points(),
              this->get_mapping(),
              *ascii_profile_reference_mapping));
        }

      MaterialModel::Interface<dim>::fill_additional_material_model_inputs(input,
                                                                           solution,
                                                                           fe_values,
                                                                           introspection);
    }



    template <int dim>
    void
    Viscoelastic<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      elastic_rheology.create_elastic_additional_outputs(out);
    }

    template <int dim>
    bool
    Viscoelastic<dim>::use_instantaneous_elastic_response_at_timestep_zero() const
    {
      return elastic_rheology.get_use_instantaneous_elastic_response_at_timestep_zero();
    }



    template <int dim>
    double
    Viscoelastic<dim>::fixed_elastic_time_step() const
    {
      return elastic_rheology.get_fixed_elastic_time_step();
    }



    template <int dim>
    double
    Viscoelastic<dim>::initial_elastic_time_step() const
    {
      return elastic_rheology.initial_elastic_time_step();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(Viscoelastic,
                                   "viscoelastic",
                                   "An implementation of a simple linear viscoelastic rheology that "
                                   "includes deviatoric elasticity by default. The default rheology takes "
                                   "into account elastic shear strength (e.g., shear modulus) and remains "
                                   "incompressible. The default-off `Enable compressible Maxwell' option "
                                   "additionally supplies a finite elastic bulk modulus to the `elastic "
                                   "pressure evolution' mass-conservation formulation; it does not make "
                                   "the physical material density pressure-dependent. This optional path "
                                   "supports assembled AMG/direct solvers and local-smoothing block GMG, "
                                   "while global-coarsening GMG remains explicitly rejected. PREM/VM5a "
                                   "use still requires scientific validation against canonical CitcomSVE "
                                   "3.0 on G2 and is not yet production-ready. The model allows specifying an "
                                   "arbitrary number "
                                   "of compositional fields, where each field represents a different "
                                   "rock type or component of the viscoelastic stress tensor. The stress "
                                   "tensor in 2d and 3d, respectively, contains 3 or 6 components. The "
                                   "compositional fields representing these components must be named "
                                   "and listed in a very specific format, which is designed to minimize "
                                   "mislabeling stress tensor components as distinct 'compositional "
                                   "rock types' (or vice versa). For 2d models, the first six compositional "
                                   "fields of type stress must be labeled 've\\_stress\\_xx', 've\\_stress\\_yy' "
                                   "and 've\\_stress\\_xy', 've\\_stress\\_xx\\_old', 've\\_stress\\_yy\\_old' "
                                   "and 've\\_stress\\_xy\\_old', In 3d, the first twelve compositional fields "
                                   "of type stress must be labeled 've\\_stress\\_xx', 've\\_stress\\_yy', "
                                   "'ve\\_stress\\_zz', 've\\_stress\\_xy', 've\\_stress\\_xz', 've\\_stress\\_yz', "
                                   "'ve\\_stress\\_xx\\_old', 've\\_stress\\_yy\\_old', 've\\_stress\\_zz\\_old',  "
                                   "'ve\\_stress\\_xy\\_old', 've\\_stress\\_xz\\_old', 've\\_stress\\_yz\\_old'. "
                                   "\n\n "
                                   "Expanding the model to include non-linear viscous flow (e.g., "
                                   "diffusion/dislocation creep) and plasticity would produce a "
                                   "constitutive relationship commonly referred to as partial "
                                   "elastoviscoplastic (e.g., pEVP) in the geodynamics community. "
                                   "While extensively discussed and applied within the geodynamics "
                                   "literature, notable references include: "
                                   "Moresi et al. (2003), J. Comp. Phys., v. 184, p. 476-497. "
                                   "Gerya and Yuen (2007), Phys. Earth. Planet. Inter., v. 163, p. 83-105. "
                                   "Gerya (2010), Introduction to Numerical Geodynamic Modeling. "
                                   "Kaus (2010), Tectonophysics, v. 484, p. 36-47. "
                                   "Choi et al. (2013), J. Geophys. Res., v. 118, p. 2429-2444. "
                                   "Keller et al. (2013), Geophys. J. Int., v. 195, p. 1406-1442. "
                                   "\n\n "
                                   "The overview below directly follows Moresi et al. (2003) eqns. 23-32. "
                                   "However, an important distinction between this material model and "
                                   "the studies above is the use of compositional fields, rather than "
                                   "particles, to track individual components of the viscoelastic stress "
                                   "tensor. The material model will be updated when an option to track "
                                   "and calculate viscoelastic stresses with particles is implemented. "
                                   "\n\n "
                                   "Moresi et al. (2003) begins (eqn. 23) by writing the deviatoric "
                                   "rate of deformation ($\\hat{D}$) as the sum of elastic "
                                   "($\\hat{D_{e}}$) and viscous ($\\hat{D_{v}}$) components: "
                                   "$\\hat{D} = \\hat{D_{e}} + \\hat{D_{v}}$.  "
                                   "These terms further decompose into "
                                   "$\\hat{D_{v}} = \\frac{\\tau}{2\\eta}$ and "
                                   "$\\hat{D_{e}} = \\frac{\\overset{\\nabla}{\\tau}}{2\\mu}$, where "
                                   "$\\tau$ is the viscous deviatoric stress, $\\eta$ is the shear viscosity, "
                                   "$\\mu$ is the shear modulus and $\\overset{\\nabla}{\\tau}$ is the "
                                   "Jaumann corotational stress rate. This later term (eqn. 24) contains the "
                                   "time derivative of the deviatoric stress ($\\dot{\\tau}$) and terms that "
                                   "account for material spin (e.g., rotation) due to advection: "
                                   "$\\overset{\\nabla}{\\tau} = \\dot{\\tau} + {\\tau}W -W\\tau$. "
                                   "Above, $W$ is the material spin tensor (eqn. 25): "
                                   "$W_{ij} = \\frac{1}{2} \\left (\\frac{\\partial V_{i}}{\\partial x_{j}} - "
                                   "\\frac{\\partial V_{j}}{\\partial x_{i}} \\right )$. "
                                   "\n\n "
                                   "The Jaumann stress-rate can also be approximated using terms from the "
                                   "previous time step ($t$) and current time step ($t + \\Delta t^{e}$): "
                                   "$\\smash[t]{\\overset{\\nabla}{\\tau}}^{t + \\Delta t^{e}} \\approx "
                                   "\\frac{\\tau^{t + \\Delta t^{e} - \\tau^{t}}}{\\Delta t^{e}} - "
                                   "W^{t}\\tau^{t} + \\tau^{t}W^{t}$. "
                                   "In this material model, the size of the time step above ($\\Delta t^{e}$) "
                                   "can be specified as the numerical time step size or an independent fixed time "
                                   "step. If the latter case is selected, a linear interpolation will be applied"
                                   "to account for the differences between the numerical "
                                   "and fixed elastic time step (eqn. 32). If one selects to use a fixed elastic time "
                                   "step throughout the model run, this can still be achieved by using CFL and "
                                   "maximum time step values that restrict the numerical time step to a specific time."
                                   "\n\n "
                                   "The formulation above allows rewriting the total deviatoric stress (eqn. 29) as\n "
                                   "$\\tau^{t + \\Delta t^{e}} = \\eta_\\text{eff} \\left ( "
                                   "2\\hat{D}^{t + \\triangle t^{e}} + \\frac{\\tau^{t}}{\\mu \\Delta t^{e}} + "
                                   "\\frac{W^{t}\\tau^{t} - \\tau^{t}W^{t}}{\\mu}  \\right )$. "
                                   "\n\n "
                                   "The effective viscosity (eqn. 28) is a function of the viscosity ($\\eta$), "
                                   "elastic time step size ($\\Delta t^{e}$) and shear relaxation time "
                                   "($ \\alpha = \\frac{\\eta}{\\mu} $): "
                                   "$\\eta_\\text{eff} = \\eta \\frac{\\Delta t^{e}}{\\Delta t^{e} + \\alpha}$ "
                                   "The magnitude of the shear modulus thus controls how much the effective "
                                   "viscosity is reduced relative to the initial viscosity. "
                                   "\n\n "
                                   "Elastic effects are introduced into the governing Stokes equations through "
                                   "an elastic force term (eqn. 30 updated to the term in eqn. 5 in Farrington et al. 2014) "
                                   "using stresses from the previous time step rotated and advected into the current time step: "
                                   "$F^{e,t} = -\\frac{\\eta_\\text{eff}}{\\mu \\Delta t^{e}} \\tau^{0adv}$. "
                                   "This force term is added onto the right-hand side force vector in the "
                                   "system of equations. "
                                   "\n\n "
                                   "The value of each compositional field representing distinct rock types at a "
                                   "point is interpreted to be a volume fraction of that rock type. If the sum of "
                                   "the compositional field volume fractions is less than one, then the remainder "
                                   "of the volume is assumed to be 'background material'."
                                   "\n\n "
                                   "Several model parameters (densities, elastic shear moduli, thermal expansivities, "
                                   "thermal conductivies, specific heats) can be defined per-compositional field. "
                                   "For each material parameter the user supplies a comma delimited list of length "
                                   "N+1, where N is the number of compositional fields. The additional field corresponds "
                                   "to the value for background material. They should be ordered ''background, "
                                   "composition1, composition2...''. However, the first 3 (2d) or 6 (3d) composition "
                                   "fields correspond to components of the elastic stress tensor and their material "
                                   "values will not contribute to the volume fractions. If a single value is given, then "
                                   "all the compositional fields are given that value. Other lengths of lists are not "
                                   "allowed. For a given compositional field the material parameters are treated as "
                                   "constant, except density, which varies linearly with temperature according to the "
                                   "thermal expansivity. "
                                   "\n\n "
                                   "When more than one compositional field is present at a point, they are averaged "
                                   "arithmetically. An exception is viscosity, which may be averaged arithmetically, "
                                   "harmonically, geometrically, or by selecting the viscosity of the composition field "
                                   "with the greatest volume fraction.")
  }
}
