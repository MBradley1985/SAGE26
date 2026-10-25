"""Cache the microUchuu Snap_48 (z = 0.022) fields used by the gas/metals investigation."""
import glob, h5py, numpy as np, os, sys
SP = sys.argv[1] if len(sys.argv) > 1 else '.'
FIELDS = ['Type','StellarMass','BulgeMass','ColdGas','MetalsColdGas','MetalsStellarMass','H2gas','H1gas',
          'SfrDisk','SfrBulge','DiskRadius','Vvir','Vmax','Mvir','Rvir','HotGas','MetalsHotGas','CGMgas','MetalsCGMgas',
          'EjectedMass','MetalsEjectedMass','IntraClusterStars','MetalsIntraClusterStars','MassLoading','Regime',
          'GalaxyIndex','CentralGalaxyIndex','OutflowRate','H2DepletionTime_Gyr','CoolingRate','FFBRegime']
def load(snap):
    out = {k: [] for k in FIELDS}
    for f in sorted(glob.glob('output/microuchuu/model_*.hdf5')):
        with h5py.File(f, 'r') as h:
            g = h[snap]
            for k in FIELDS:
                out[k].append(g[k][:])
    return {k: np.concatenate(v) for k, v in out.items()}
for snap in ['Snap_48', 'Snap_30', 'Snap_20']:
    d = load(snap)
    np.savez(os.path.join(SP, f'micro_{snap}.npz'), **d)
    print(snap, len(d['Type']))
