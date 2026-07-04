/*
 * core_simulation.h -- halo_data struct (on-disk lhalo record).
 *
 * Defines the binary layout of a single halo as stored in lhalo-format
 * merger tree files.  All tree readers cast their on-disk bytes to this
 * struct.  Field order must match the on-disk layout exactly.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct halo_data
{
    // merger tree pointers
    int Descendant;
    int FirstProgenitor;
    int NextProgenitor;
    int FirstHaloInFOFgroup;
    int NextHaloInFOFgroup;

    // properties of halo
    int Len;          // number of simulation particles bound to the (sub)halo
    float M_Mean200;  // M200 wrt mean density [10^10 Msun/h]
    union {
        float Mvir;
        float M200c;// for Millennium, Mvir=M_Crit200 [10^10 Msun/h]
    };
    float M_TopHat;   // tophat-overdensity virial mass [10^10 Msun/h]
    float Pos[3];     // comoving position [Mpc/h]
    float Vel[3];     // peculiar velocity [km/s]
    float VelDisp;    // 1-D velocity dispersion [km/s]
    float Vmax;       // maximum circular velocity [km/s]
    float Spin[3];    // specific angular momentum [Mpc/h km/s]; |Spin|/(sqrt(2) Vvir Rvir) gives the Bullock+01 spin parameter
    long long MostBoundID;  // for LHaloTrees, this is the ID of the most bound particle; for other mergertree codes, let this contain a unique haloid

    // original position in simulation tree files
    int SnapNum;
    int FileNr;
    int SubhaloIndex;
    float SubHalfMass;  // mass within the half-mass radius [10^10 Msun/h]
};

#ifdef __cplusplus
}
#endif
