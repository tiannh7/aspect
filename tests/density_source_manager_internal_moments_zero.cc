#include <aspect/density_source_manager.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class DensitySourceManagerInternalMomentsZero : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &) override
        {
          const auto &manager = this->get_density_source_manager();
          const typename DensitySourceManager<dim>::InternalMassMoments moments =
            manager.compute_internal_mass_moments(3300.0);

          std::ostringstream output;
          output << std::scientific << std::setprecision(6)
                 << "enabled="
                 << (manager.internal_density_anomalies_are_enabled("auto")
                     ? "true" : "false")
                 << ", dipole=("
                 << moments.mass_dipole[0] << ","
                 << moments.mass_dipole[1] << ","
                 << moments.mass_dipole[2] << ")"
                 << ", products=("
                 << moments.inertia_tensor[0][1] << ","
                 << moments.inertia_tensor[0][2] << ","
                 << moments.inertia_tensor[1][2] << ")";

          return {"Internal mass moments:", output.str()};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(DensitySourceManagerInternalMomentsZero,
                                  "density source manager internal moments",
                                  "Test the zero-volume internal mass-moment control.")
  }
}
