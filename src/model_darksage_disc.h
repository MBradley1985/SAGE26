/*
 * model_darksage_disc.h -- public interface for disc annulus radii (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_disc.c */
    extern void update_disc_radii(const int p, struct GALAXY *galaxies, const struct halo_data *halos,
                                  const struct params *run_params);
    extern double darksage_ics_scale_radius(const double Rvir, const double R_av);
    extern void darksage_add_ics_radius(struct GALAXY *c, const double m, const double r);
    extern double darksage_satellite_mass(const struct GALAXY *s);
    extern double darksage_satellite_radius(const struct GALAXY *c, const struct GALAXY *s, const struct params *run_params);
    extern double darksage_hot_density(const struct GALAXY *c, const double r, const struct params *run_params);
    extern double darksage_host_mass_within(const struct GALAXY *c, const double r, const double m_sat,
                                            const struct params *run_params);
    extern void darksage_project_disc(const float DiscMass[N_BINS], const double cos_angle, double NewDisc[N_BINS],
                                      const struct params *run_params, const double Rvir, const double Vvir);
    extern void darksage_precess_gas(const int p, const double dt, struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif
