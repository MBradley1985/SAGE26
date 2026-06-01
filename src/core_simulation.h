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
    int Len;
    float M_Mean200;
    union {
        float Mvir;
        float M200c;// for Millennium, Mvir=M_Crit200
    };
    float M_TopHat;
    float Pos[3];
    float Vel[3];
    float VelDisp;
    float Vmax;
    float Spin[3];
    long long MostBoundID;  // for LHaloTrees, this is the ID of the most bound particle; for other mergertree codes, let this contain a unique haloid

    // original position in simulation tree files
    int SnapNum;
    int FileNr;
    int SubhaloIndex;
    float SubHalfMass;
};

#ifdef __cplusplus
}
#endif
