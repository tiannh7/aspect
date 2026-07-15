/*
  Copyright (C) 2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/density_source_manager.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator/assemblers/stokes.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace aspect
{
  template <int dim>
  void
  report_internal_density_jump_assembler(
    const SimulatorAccess<dim> &,
    Assemblers::Manager<dim> &assemblers)
  {
    bool registered = false;
    for (const auto &assembler : assemblers.stokes_system)
      if (dynamic_cast<Assemblers::StokesInternalDensityJumpRestoring<dim> *>(
            assembler.get()) != nullptr)
        registered = true;

    std::cout << "Internal density jump restoring assembler: "
              << (registered ? "registered" : "not registered")
              << std::endl;
  }



  template <int dim>
  void
  signal_connector(SimulatorSignals<dim> &signals)
  {
    signals.set_assemblers.connect(
      &report_internal_density_jump_assembler<dim>);
  }



  namespace Postprocess
  {
    template <int dim>
    class DensitySourceManagerTabulatedRadial : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &/*statistics*/) override
        {
          AssertThrow(dim == 2, ExcNotImplemented());

          const auto point_at_radius = [](const double radius)
          {
            Point<dim> point;
            point[0] = radius;
            return point;
          };

          const auto &manager = this->get_density_source_manager();
          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << "rho "
                 << manager.reference_density(point_at_radius(0.5)) << " / "
                 << manager.reference_density(point_at_radius(1.25)) << " / "
                 << manager.reference_density(point_at_radius(1.75)) << " / "
                 << manager.reference_density(point_at_radius(2.5))
                 << ", scale " << manager.get_reference_density_scale()
                 << ", default mechanical g "
                 << manager.mechanical_gravity_magnitude(
                   point_at_radius(1.25),
                   9.81);

          if (manager.has_internal_density_jumps())
            output << ", moved-face jump "
                   << manager.internal_density_jump_across_face(
                     point_at_radius(1.25),
                     point_at_radius(1.75),
                     1.6);

          return {"Radial reference density:", output.str()};
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(DensitySourceManagerTabulatedRadial,
                                  "density source manager tabulated radial",
                                  "Test piecewise-linear tabulated radial reference density and endpoint clamping.")
  }
}

ASPECT_REGISTER_SIGNALS_CONNECTOR(aspect::signal_connector<2>,
                                  aspect::signal_connector<3>)
