#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_allvars.h"

/* Dust production from star formation — simplified model (MetalYieldsOn=0) */
extern void produce_dust(const double stars, const double metallicity, const double dt,
                         const int p, const int centralgal, struct GALAXY *galaxies,
                         const struct params *run_params);

/* Dust production from yield tables — full model (MetalYieldsOn=1, requires GSL) */
#ifdef GSL_FOUND
extern void produce_metals_dust(const double metallicity, const double dt,
                                const int p, const int centralgal,
                                struct GALAXY *galaxies, const struct params *run_params);
#endif

/* ISM grain growth / dust accretion (Asano+13 eq. 20) */
extern void accrete_dust(const double metallicity, const double dt, const int p,
                         struct GALAXY *galaxies, const struct params *run_params);

/* Dust destruction by SN shocks (Asano+13 eqs. 12, 14) */
extern void destruct_dust(const double metallicity, const double stars, const double dt,
                          const int p, struct GALAXY *galaxies, const struct params *run_params);

/* Thermal sputtering in hot gas (Popping+17 section 3.5) */
extern void dust_thermal_sputtering(const int gal, const double dt,
                                    struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif
