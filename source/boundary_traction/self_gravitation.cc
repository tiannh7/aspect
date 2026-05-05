/*
  Copyright (C) 2024 by the authors of the ASPECT code.

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

#include <aspect/boundary_traction/self_gravitation.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/simulator.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace BoundaryTraction
  {
    template <int dim>
    void
    SelfGravitation<dim>::initialize()
    {
      if (dim == 3)
        sh_transform = std::make_unique<Utilities::SphericalHarmonicTransform>(
                         max_degree, min_degree);
      else
        fourier_transform = std::make_unique<Utilities::FourierTransform>(
                              max_degree, min_degree);
    }


    template <int dim>
    void
    SelfGravitation<dim>::update()
    {
      compute_self_gravity_correction();
    }


    template <int dim>
    void
    SelfGravitation<dim>::compute_self_gravity_correction()
    {
      AssertThrow(Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(
                    this->get_geometry_model()),
                  ExcMessage("Self-gravitation requires a spherical shell geometry."));

      const GeometryModel::SphericalShell<dim> &geometry =
        Plugins::get_plugin_as_type<const GeometryModel::SphericalShell<dim>>(
          this->get_geometry_model());

      const double outer_radius = geometry.outer_radius();
      const double inner_radius = geometry.inner_radius();
      const double radius_ratio = inner_radius / outer_radius;
      const types::boundary_id top_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
      const types::boundary_id bottom_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("bottom");

      // Step 1: Collect surface and CMB topography at quadrature points
      const unsigned int quadrature_degree =
        this->introspection().polynomial_degree.temperature;
      const QGauss<dim - 1> quadrature_formula_face(quadrature_degree);

      FEFaceValues<dim> fe_face_values(this->get_mapping(),
                                       this->get_fe(),
                                       quadrature_formula_face,
                                       update_values |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

      const double delta_rho_surf = density_below_surface - density_above_surface;

      // Surface topography data
      std::vector<double> phi_pts;
      std::vector<double> theta_pts; // only used in 3D
      std::vector<double> weight_pts;
      std::vector<double> topo_pts;

      // CMB topography data
      std::vector<double> cmb_phi_pts;
      std::vector<double> cmb_theta_pts; // only used in 3D
      std::vector<double> cmb_weight_pts;
      std::vector<double> cmb_topo_pts;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            for (const unsigned int f : cell->face_indices())
              {
                if (!cell->at_boundary(f))
                  continue;

                const types::boundary_id bid = cell->face(f)->boundary_id();
                const bool is_top    = (bid == top_boundary_id);
                const bool is_bottom = (bid == bottom_boundary_id) && include_cmb_contribution;

                if (!is_top && !is_bottom)
                  continue;

                fe_face_values.reinit(cell, f);

                for (unsigned int q = 0;
                     q < fe_face_values.n_quadrature_points;
                     ++q)
                  {
                    const Point<dim> position =
                      fe_face_values.quadrature_point(q);

                    const std::array<double, dim> scoord =
                      aspect::Utilities::Coordinates::
                      cartesian_to_spherical_coordinates(position);

                    // scoord: 2D = {r, phi}, 3D = {r, phi, theta}
                    const double ph = scoord[1]; // longitude / azimuthal angle

                    if (is_top)
                      {
                        const double h_rock =
                          this->get_geometry_model()
                          .height_above_reference_surface(position);

                        // Compute the external load's equivalent height.
                        // The total traction from the boundary traction manager
                        // includes all plugins (ascii data + our old correction).
                        // Subtract our own old contribution to isolate the load.
                        const Tensor<1,dim> face_normal =
                          fe_face_values.normal_vector(q);

                        const Tensor<1,dim> total_traction =
                          this->get_boundary_traction_manager()
                          .boundary_traction(top_boundary_id,
                                             position,
                                             face_normal);

                        const Tensor<1,dim> our_old_traction =
                          this->boundary_traction(top_boundary_id,
                                                  position,
                                                  face_normal);

                        const Tensor<1,dim> load_traction =
                          total_traction - our_old_traction;

                        // Inward load traction (T·n < 0) → positive surface mass
                        // σ_load = -T_load·n / g,  h_load = σ_load / Δρ
                        const double g_magnitude =
                          this->get_gravity_model().gravity_vector(position).norm();

                        double h_load = 0.0;
                        if (g_magnitude > 0 && delta_rho_surf > 0)
                          h_load = -(load_traction * face_normal) /
                                   (delta_rho_surf * g_magnitude);

                        const double h_effective = h_rock + h_load;

                        const double ref_radius = outer_radius;
                        const double w =
                          fe_face_values.JxW(q) /
                          (dim == 3 ? ref_radius * ref_radius : ref_radius);

                        phi_pts.push_back(ph);
                        if (dim == 3)
                          theta_pts.push_back(scoord[2]);
                        weight_pts.push_back(w);
                        topo_pts.push_back(h_effective);
                      }
                    else // is_bottom
                      {
                        const double r = scoord[0];
                        const double cmb_topography = r - inner_radius;
                        const double ref_radius = inner_radius;
                        const double w =
                          fe_face_values.JxW(q) /
                          (dim == 3 ? ref_radius * ref_radius : ref_radius);

                        cmb_phi_pts.push_back(ph);
                        if (dim == 3)
                          cmb_theta_pts.push_back(scoord[2]);
                        cmb_weight_pts.push_back(w);
                        cmb_topo_pts.push_back(cmb_topography);
                      }
                  }
              }
          }

      // Step 2 & 3: SH/Fourier analysis + self-gravity kernel
      //
      // 3D self-gravity ratio: Rsg(l) = 3*delta_rho / ((2l+1)*rho_mean)
      // 2D self-gravity ratio: Rsg(n) = 2*delta_rho / (n * rho_mean)  [n>=1]
      //   (For n=0, Rsg=0 since uniform mass shift does not change the potential gradient.)
      //
      // CMB scaling: 3D: (r_cmb/R)^(l+2),  2D: (r_cmb/R)^(n+1)

      const double delta_rho_cmb  = density_below_cmb - density_above_cmb;

      if (dim == 3)
        {
          // --- 3D: Spherical Harmonic path ---
          auto [cos_topo, sin_topo] = sh_transform->analyze(
                                        theta_pts, phi_pts, weight_pts, topo_pts,
                                        this->get_mpi_communicator());

          const unsigned int n_coeff = sh_transform->n_coefficients();

          std::vector<double> surf_filter(max_degree + 1, 0.0);
          for (unsigned int l = min_degree; l <= max_degree; ++l)
            surf_filter[l] = 3.0 * delta_rho_surf /
                             ((2.0 * l + 1.0) * planet_mean_density);

          correction_cos_coeffs = cos_topo;
          correction_sin_coeffs = sin_topo;
          sh_transform->apply_degree_filter(correction_cos_coeffs,
                                            correction_sin_coeffs,
                                            surf_filter);

          if (include_cmb_contribution && !cmb_topo_pts.empty())
            {
              auto [cos_cmb, sin_cmb] = sh_transform->analyze(
                                          cmb_theta_pts, cmb_phi_pts,
                                          cmb_weight_pts, cmb_topo_pts,
                                          this->get_mpi_communicator());

              std::vector<double> cmb_filter(max_degree + 1, 0.0);
              for (unsigned int l = min_degree; l <= max_degree; ++l)
                cmb_filter[l] = 3.0 * delta_rho_cmb /
                                ((2.0 * l + 1.0) * planet_mean_density)
                                * std::pow(radius_ratio, static_cast<int>(l) + 2);

              sh_transform->apply_degree_filter(cos_cmb, sin_cmb, cmb_filter);

              for (unsigned int i = 0; i < n_coeff; ++i)
                {
                  correction_cos_coeffs[i] += cos_cmb[i];
                  correction_sin_coeffs[i] += sin_cmb[i];
                }
            }
        }
      else
        {
          // --- 2D: Fourier path ---
          auto [cos_topo, sin_topo] = fourier_transform->analyze(
                                        phi_pts, weight_pts, topo_pts,
                                        this->get_mpi_communicator());

          const unsigned int n_coeff = fourier_transform->n_coefficients();

          std::vector<double> surf_filter(max_degree + 1, 0.0);
          for (unsigned int n = std::max(min_degree, 1u); n <= max_degree; ++n)
            surf_filter[n] = 2.0 * delta_rho_surf /
                             (static_cast<double>(n) * planet_mean_density);

          correction_cos_coeffs = cos_topo;
          correction_sin_coeffs = sin_topo;
          fourier_transform->apply_degree_filter(correction_cos_coeffs,
                                                 correction_sin_coeffs,
                                                 surf_filter);

          if (include_cmb_contribution && !cmb_topo_pts.empty())
            {
              auto [cos_cmb, sin_cmb] = fourier_transform->analyze(
                                          cmb_phi_pts, cmb_weight_pts, cmb_topo_pts,
                                          this->get_mpi_communicator());

              std::vector<double> cmb_filter(max_degree + 1, 0.0);
              for (unsigned int n = std::max(min_degree, 1u); n <= max_degree; ++n)
                cmb_filter[n] = 2.0 * delta_rho_cmb /
                                (static_cast<double>(n) * planet_mean_density)
                                * std::pow(radius_ratio, static_cast<int>(n) + 1);

              fourier_transform->apply_degree_filter(cos_cmb, sin_cmb, cmb_filter);

              for (unsigned int i = 0; i < n_coeff; ++i)
                {
                  correction_cos_coeffs[i] += cos_cmb[i];
                  correction_sin_coeffs[i] += sin_cmb[i];
                }
            }
        }

      // Cache for later point queries
      cached_phi = phi_pts;
      if (dim == 3)
        cached_theta = theta_pts;
    }


    template <int dim>
    Tensor<1, dim>
    SelfGravitation<dim>::boundary_traction(
      const types::boundary_id /*boundary_indicator*/,
      const Point<dim> &position,
      const Tensor<1, dim> &normal_vector) const
    {
      // Convert position to spherical coordinates
      const std::array<double, dim> scoord =
        aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(
          position);
      const double ph = scoord[1]; // longitude / azimuthal angle

      // If no coefficients have been computed yet (first timestep),
      // return zero traction.
      if (correction_cos_coeffs.empty())
        return Tensor<1, dim>();

      // Synthesize the self-gravity correction at this point
      double correction_value = 0.0;
      if (dim == 3)
        {
          const double th = scoord[2]; // colatitude
          const std::vector<double> th_vec = {th};
          const std::vector<double> ph_vec = {ph};
          const std::vector<double> correction =
            sh_transform->synthesize(correction_cos_coeffs,
                                     correction_sin_coeffs,
                                     th_vec, ph_vec);
          correction_value = correction[0];
        }
      else
        {
          const std::vector<double> ph_vec = {ph};
          const std::vector<double> correction =
            fourier_transform->synthesize(correction_cos_coeffs,
                                          correction_sin_coeffs,
                                          ph_vec);
          correction_value = correction[0];
        }

      // The correction traction is outward-pointing (reducing the load):
      //   T = delta_rho_surf * g * correction(position)
      const Tensor<1, dim> gravity =
        this->get_gravity_model().gravity_vector(position);
      const double g_magnitude = gravity.norm();
      const double delta_rho_surf = density_below_surface - density_above_surface;

      return delta_rho_surf * g_magnitude * correction_value * normal_vector;
    }


    template <int dim>
    void
    SelfGravitation<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Self gravitation");
        {
          prm.declare_entry("Maximum degree", "40",
                            Patterns::Integer(0),
                            "Maximum spherical harmonic degree for the "
                            "self-gravitation calculation.");

          prm.declare_entry("Minimum degree", "0",
                            Patterns::Integer(0),
                            "Minimum spherical harmonic degree for the "
                            "self-gravitation calculation.");

          prm.declare_entry("Density above surface", "0",
                            Patterns::Double(0),
                            "Density immediately above the deformed surface "
                            "boundary in kg/m^3. For a free surface in vacuum "
                            "or thin atmosphere, set to 0. For a seafloor "
                            "under ocean, set to water density (e.g., 1030).");

          prm.declare_entry("Density below surface", "3500",
                            Patterns::Double(0),
                            "Density immediately below the deformed surface "
                            "boundary in kg/m^3. For rock topography, use "
                            "crustal density (e.g., 3500). For an ice cap "
                            "sitting on rock, use ice density (e.g., 917).");

          prm.declare_entry("Density above cmb", "5500",
                            Patterns::Double(0),
                            "Density immediately above the CMB (lower mantle side) "
                            "in kg/m^3. Earth: ~5500, Mars: ~3800.");

          prm.declare_entry("Density below cmb", "9900",
                            Patterns::Double(0),
                            "Density immediately below the CMB (outer core side) "
                            "in kg/m^3. Earth: ~9900, Mars: ~6200.");

          prm.declare_entry("Planet mean density", "5515",
                            Patterns::Double(0),
                            "Mean density of the planet in kg/m^3. "
                            "Earth: 5515, Mars: 3390.");

          prm.declare_entry("Include cmb contribution", "true",
                            Patterns::Bool(),
                            "Whether to include the CMB topography contribution "
                            "to the self-gravitational potential perturbation. "
                            "Set to false if only surface topography feedback is needed.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    SelfGravitation<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary traction model");
      {
        prm.enter_subsection("Self gravitation");
        {
          max_degree = prm.get_integer("Maximum degree");
          min_degree = prm.get_integer("Minimum degree");
          density_above_surface = prm.get_double("Density above surface");
          density_below_surface = prm.get_double("Density below surface");
          density_above_cmb = prm.get_double("Density above cmb");
          density_below_cmb = prm.get_double("Density below cmb");
          planet_mean_density = prm.get_double("Planet mean density");
          include_cmb_contribution = prm.get_bool("Include cmb contribution");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}


// Explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(
      SelfGravitation,
      "self gravitation",
      "A boundary traction model that computes the self-gravitational "
      "feedback from surface topography. When a surface load deforms the "
      "planet, the resulting topography changes the gravitational potential, "
      "which in turn modifies the effective surface load. "
      "\n\n"
      "The density contrast at the surface is "
      "$\\Delta\\rho = \\rho_{below\\_surface} - \\rho_{above\\_surface}$. "
      "For spherical harmonic degree $l$, the self-gravity ratio is: "
      "$\\mathrm{self\\_gravity\\_ratio}(l) = 3 \\Delta\\rho / ((2l+1) \\bar{\\rho})$. "
      "This reduces the effective load by a factor "
      "$(1 - \\mathrm{self\\_gravity\\_ratio}(l))$. "
      "\n\n"
      "This plugin computes the topography from the mesh deformation "
      "(free surface), expands it in spherical harmonics, applies the "
      "degree-dependent self-gravity kernel, and applies the resulting "
      "correction as an outward normal traction on the surface boundary.")
  }
}
