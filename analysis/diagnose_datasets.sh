#!/bin/bash
# Dataset diagnostics — run from /project/etp4/L.Chrusciel/bachelor-thesis

echo "=== 1. Layer-IDs pro Datensatz ===" && python -c "
import uproot, numpy as np, awkward as ak
for fpath in ['/project/etp4/L.Chrusciel/Datensatz/Data_for_Roman_29deg_530V_100ns_x9.root','/project/etp4/L.Chrusciel/Datensatz/H4_GIF/Data_For_Lars_29deg_att0_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H4_GIF/Data_For_Lars_29deg_att0_530V_200ns.root','/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_29deg_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_20deg_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_15deg_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_10deg_530V_100ns.root']:
    t = uproot.open(fpath)['ana']
    layers = np.asarray(ak.flatten(t['out_layer'].array(entry_stop=2000, library='ak')))
    vals, counts = np.unique(layers.astype(int), return_counts=True)
    print(fpath.split('/')[-1])
    [print(f'  layer {v}: {c} strips ({100*c/len(layers):.1f}%)') for v, c in zip(vals, counts)]
    print()
"

echo "=== 2. n_strips pro Layer 3 ===" && python -c "
import uproot, numpy as np, awkward as ak
for fpath in ['/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_29deg_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H4_GIF/Data_For_Lars_29deg_att0_530V_100ns.root']:
    t = uproot.open(fpath)['ana']
    xpos = t['out_xpos'].array(entry_stop=5000, library='ak')
    layers = t['out_layer'].array(entry_stop=5000, library='ak')
    ns = np.array([int((np.asarray(layers[i])==3).sum()) for i in range(len(xpos))])
    ns = ns[ns>0]
    print(f'{fpath.split(\"/\")[-1]}  layer=3')
    print(f'  n_strips: mean={ns.mean():.1f}  median={np.median(ns):.0f}  max={ns.max()}')
    print()
"

echo "=== 3. Zeitverteilung layer 3 ===" && python -c "
import uproot, numpy as np, awkward as ak
for fpath in ['/project/etp4/L.Chrusciel/Datensatz/Data_for_Roman_29deg_530V_100ns_x9.root','/project/etp4/L.Chrusciel/Datensatz/H4_GIF/Data_For_Lars_29deg_att0_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H4_GIF/Data_For_Lars_29deg_att0_530V_200ns.root','/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_29deg_530V_100ns.root']:
    t = uproot.open(fpath)['ana']
    times = t['out_time'].array(entry_stop=5000, library='ak')
    layers = t['out_layer'].array(entry_stop=5000, library='ak')
    all_t = [float(v) for i in range(len(times)) for v in np.asarray(times[i])[np.asarray(layers[i])==3]]
    ft = np.array(all_t); ft = ft[np.isfinite(ft)]
    print(f'{fpath.split(\"/\")[-1]}  layer=3')
    print(f'  t: mean={ft.mean():.1f}  std={ft.std():.1f}  min={ft.min():.1f}  max={ft.max():.1f} ns')
    print()
"

echo "=== 4. x-Position und track_icept layer 3 ===" && python -c "
import uproot, numpy as np, awkward as ak
for fpath in ['/project/etp4/L.Chrusciel/Datensatz/Data_for_Roman_29deg_530V_100ns_x9.root','/project/etp4/L.Chrusciel/Datensatz/H8_angularScan/H8_29deg_530V_100ns.root','/project/etp4/L.Chrusciel/Datensatz/H4_GIF/Data_For_Lars_29deg_att0_530V_100ns.root']:
    t = uproot.open(fpath)['ana']
    xpos = t['out_xpos'].array(entry_stop=5000, library='ak')
    layers = t['out_layer'].array(entry_stop=5000, library='ak')
    icept = t['out_track_icept'].array(entry_stop=5000, library='np')
    all_x = [float(v) for i in range(len(xpos)) for v in np.asarray(xpos[i])[np.asarray(layers[i])==3]]
    fx = np.array(all_x)
    print(f'{fpath.split(\"/\")[-1]}  layer=3')
    print(f'  x:     mean={fx.mean():.1f}  std={fx.std():.1f}  min={fx.min():.1f}  max={fx.max():.1f} mm')
    print(f'  icept: mean={icept.mean():.1f}  std={icept.std():.1f} mm')
    print()
"
