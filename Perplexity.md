# GNN Position Reconstruction — Research Log

Dieser Log dokumentiert alle Experimente, Entscheidungen, Diskussionen und Findings
chronologisch. Ziel: reproduzierbare wissenschaftliche Dokumentation für die Bachelorarbeit.

---

## Metrik-Definitionen

| Metrik | Definition | Verwendung |
|---|---|---|
| σ_core | Breite der engen Gauß-Komponente im Double-Gaussian-Fit auf \|residual\| < 2mm | Fair comparison mit Vogel (TC-Methode) |
| σ₆₈ | 0.5*(Q84−Q16), modell-frei, robust | Primäre Trainingsmetrik |
| eff_reco | Anteil Events innerhalb ±2mm / Events nach road-empty-Filter | Rekonstruktionseffizienz |
| eff_total | Anteil Events innerhalb ±2mm / alle Events | Gesamteffizienz |

---

## Feature-Set Historie

### v1 — Baseline (3 Strip-Features)
`x_norm`, `q_norm`, `t_norm`  
Global: `slope_norm`, `nonprec_norm`, `log1p(n_strips)`

### v2 — + Drifttiefe
+ `z_norm = t * v_drift`

### v3 — + muTPC-Korrektur (xcorr)
+ `x_corr_rel = (x - z*tan(θ) - anchor) / 5`  
Physikalische Grundlage: Vogel Gl. 5.40 — Verschiebung des ladungsgewichteten Schwerpunkts
proportional zur Drifttiefe und zum Neigungswinkel. Das ist das entscheidende Feature.

### v4 — xcorr_v2 (aktuell bestes Modell, Stand 2026-06-27)
**Strip-Features (6):** `x_norm`, `q_norm`, `t_norm`, `x_rel`, `z_norm`, `x_corr_rel`  
**Global-Features (3):** `slope_norm`, `nonprec_norm`, `log1p(n_strips)`

---

## Experimentelle Ergebnisse

### Training auf H4 29° (Data_for_Roman_29deg_530V_100ns_x9.root)

| Modell | Parameter | σ_core | σ₆₈ | Anmerkung |
|---|---|---|---|---|
| gnn_tc_xcorr_v1 | 893k (d=128, h=8, l=4) | 113 µm | 168 µm | großes Modell |
| gnn_tc_xcorr_small | 125k (d=64, h=4, l=2) | **109 µm** | 167 µm | besser trotz weniger Params |
| Vogel TC-Methode | — | 126 µm | — | Referenzwert aus Vogel §7.2 |

**Befund:** Das kleine Modell übertrifft Vogels TC-Methode. Feature `x_corr_rel` ist der
entscheidende Faktor — ohne es bricht die Performance ein.

---

### Zero-Shot Evaluation — gnn_tc_xcorr_v1 (2026-06-27)

Modell ohne Nachtraining auf fremde Datensätze angewendet. Layer-Filter aktiv:
H4/H4_GIF → layer 7, H8 → layer 3 (gleicher physikalischer Detektor, unterschiedliche
Nummerierung per Fabians README).

#### H8 Angular Scan

| Datensatz | θ | σ_core | σ₆₈ | eff_reco | eff_total |
|---|---|---|---|---|---|
| H8 10° | 10° | 348 µm | 513 µm | 96.1% | 95.2% |
| H8 15° | 15° | 309 µm | 558 µm | 94.1% | 93.1% |
| H8 20° | 20° | 395 µm | 540 µm | 94.2% | 93.3% |
| H8 24° | 24° | 275 µm | 548 µm | 94.3% | 93.3% |
| H8 29° | 29° | 166 µm | 618 µm | 92.7% | 90.8% |

#### H4 GIF

| Datensatz | θ | σ_core | σ₆₈ | eff_reco | eff_total |
|---|---|---|---|---|---|
| H4_GIF 530V/100ns | 29° | 163 µm | 1096 µm | 90.3% | 73.0% |
| H4_GIF 530V/200ns | 29° | 238 µm | 1051 µm | 92.2% | 75.6% |
| H4_GIF 520V/100ns | 29° | 196 µm | 1121 µm | 90.4% | 72.6% |

**Befunde:**

1. **H8:** σ_core degradiert mit Winkelabstand zu 29°. Kein monotoner Trend (20° ist schlechter
   als 15°) — deutet auf Detektorhardware-Unterschiede zwischen H8 und H4 hin.

2. **H4_GIF:** σ_core vergleichsweise gut (163–238 µm), aber σ₆₈ ~1100 µm. Scharfer Core,
   massive Tails. Vermuteter Grund: GIF-Hintergrundtreffer im Road. Das Modell rekonstruiert
   den echten Muon-Hit korrekt, aber viele Events haben unkorrellierte Hits die den Anchor
   verzerren → große Tails.

3. **Residual vs. Position — systematischer Tilt:** In den Residual-vs-Position-Plots ist bei
   kleinen Winkeln (10°, 15°) eine starke Neigung der Punktwolke sichtbar — oben links,
   unten rechts. Bei 29° (Trainingswinkel) liegt die Wolke flach. Die rote Linie (gleitender
   Mittelwert der Residuals) macht diesen positionsabhängigen Bias direkt sichtbar.

---

## Diskussion: Ursache des Tilts in Residual-vs-Position

**Lars (Beobachtung, 2026-06-27):** "Bei 15° ist die Punktwolke wirklich im Winkel drin,
bei 29° ungefähr gleich verteilt."

**Analyse:** Der Tilt entsteht nicht durch die a,b-Frame-Transformation (die wird pro
Datensatz neu gefittet). Der Grund liegt in `x_corr_rel`:

```
x_corr_rel = (x - z*tan(θ) - anchor) / 5
```

tan(29°) ≈ 0.554, tan(15°) ≈ 0.268 — Faktor ~2 Unterschied. Das Modell hat gelernt:
"wenn x_corr_rel so aussieht → Position dort." Dieses Mapping gilt nur für 29°. Bei 15°
bedeutet der gleiche x_corr_rel-Wert eine andere physikalische Position. Der Fehler ist
linear mit der Position (weil der Term z*tan(θ) über den Detektor variiert) → Tilt.

**Lars (Einwand):** "Aber x_corr_rel wird doch mit dem richtigen θ aus der Config berechnet?"

**Antwort:** Das Feature selbst stimmt. Das Problem ist, dass das Modell die Interpretation
des Features nie für andere Winkel gelernt hat. Es hat bei 29° gelernt: "x_corr_rel=X → 
Position Y." Diese Abbildung ist 29°-spezifisch, auch wenn das Feature korrekt berechnet wird.

**Lars (Einwand):** "Ist das wirklich nicht durch den a,b-Fit verursacht?"

**Antwort:** Nein. a,b transformiert Modul→Tracker-Koordinaten und wird pro Datensatz neu
gefittet. Es ist geometrisch notwendig und korrekt. Das Tilt-Problem existiert im Modul-
Koordinatensystem, bevor a,b angewendet wird.

---

## Feature Engineering v5 — Multi-Winkel-Generalisierung (2026-06-27)

### Motivation

Zero-Shot zeigt klare Winkelabhängigkeit. Für Schritt 2 der Arbeit (Modell das auf allen
Winkeln gut funktioniert) sind neue Features notwendig.

### Analyse: Was Vogel physikalisch beschreibt (Vogel_Fabian.txt)

Vogel beschreibt zwei Methoden für geneigte Tracks (§5.4):

1. **μTPC-Methode** (Gl. 5.37): Linearer Fit über (x_strip, z_strip) → Slope m_muTPC → Winkel:
   `Θ = 90° - arctan(vD / (m_muTPC * p))`
   Der Slope ist eine direkte Messung des Winkels aus den Clusterdaten selbst.

2. **TC-Methode** (Gl. 5.40): Ladungsgewichteter Schwerpunkt + Zeitkorrektur.
   `∆Y = ∆t * vD * tan(Θ)` — das ist der Ursprung von x_corr_rel.

Weitere physikalisch relevante Größen aus Vogel:
- **Cluster-Größe** (Gl. 7.1): `N_strips = tan(Θ) * Z_drift / p` — explizit winkelabhängig
- **Ladungsasymmetrie**: Bei geneigten Tracks sammelt sich Ionisation asymmetrisch (§5.4,
  Abb. 5.10). Erste vs. letzte Strips haben systematisch unterschiedliche Ladung.
- **Ladungsgewichtetes Timing** (tCW): Zeitkorrelation mit Position ist Winkel-kodiert.

### Neue Features — Entscheidung und Begründung

#### 1. θ als Global-Feature (sin/cos Encoding)
**Warum:** Das Modell bekommt den Winkel nie explizit gesagt. Alle winkelabhängigen
Zusammenhänge müssen implizit aus den Cluster-Features erschlossen werden.

**Lars (Einwand):** "Ist der nicht schon drin über x_corr_rel?"
**Antwort:** Nein. x_corr_rel nutzt tan(θ) zur Feature-Berechnung, aber der Wert von θ
selbst ist dem Modell nicht bekannt. Bei Multi-Winkel-Training sieht das Modell
x_corr_rel-Werte von verschiedenen Winkeln — ohne θ als Feature kann es sie nicht
unterscheiden.

**Encoding:** sin(θ) und cos(θ) statt rohem θ_deg — periodisch, skalierungsfrei,
und tan(θ) = sin(θ)/cos(θ) ist direkt ableitbar. Das entspricht dem Standard in der
Physik-ML-Literatur für Winkelgrößen.

#### 2. μTPC-Slope als Global-Feature
**Physik:** Vogel Gl. 5.37 — der Slope des linearen Fits über (x_strip, z_strip) kodiert
den Winkel direkt aus den Daten. Vorteil: das Modell kann den Winkel *aus den Hits selbst*
schätzen, unabhängig vom konfigurierten θ. Das ist robust gegenüber Kalibrations-
unsicherheiten.

**Berechnung:** `m = polyfit(x_cluster, z_cluster, 1)[0]`

**Lars (Diskussion):** "Ist das wirklich nötig wenn θ schon drin ist?"
**Antwort:** Ja, aus zwei Gründen: (a) Der gemessene Slope kann vom konfigurierten θ
abweichen (Kalibration, Alignment). (b) Er kodiert zusätzliche Information über die
Qualität der muTPC-Rekonstruktion im konkreten Event — ein breites, gut aufgelöstes
Cluster hat einen zuverlässigeren Slope als ein 2-Strip-Cluster.

#### 3. Ladungsasymmetrie als Global-Feature
**Physik:** Vogel §5.4, Abb. 5.10 — bei geneigten Tracks akkumuliert Ionisation asymmetrisch
entlang des Tracks. Die ersten Strips (Eintrittspunkt) haben systematisch andere Ladung als
die letzten (Austrittspunkt).

**Berechnung:** `q_asym = (q_last_half - q_first_half) / q_total`
Wobei first/last nach x-Position sortiert.

**Lars (Einwand):** "Ist das nicht schon in q_norm + z_norm kodiert?"
**Antwort:** Indirekt ja, aber das Modell muss es erst aus den per-Strip-Features extrahieren.
Ein explizites Cluster-Level-Feature gibt dem Modell diese Information direkt — ohne dass
der Attention-Mechanismus sie aggregieren muss. Es reduziert die Lernlast.

### Feature-Set v5 — Zusammenfassung

**Strip-Features (6, unverändert):**
`x_norm`, `q_norm`, `t_norm`, `x_rel`, `z_norm`, `x_corr_rel`

**Global-Features (6, erweitert von 3):**
| Feature | Berechnung | Physik |
|---|---|---|
| slope_norm | track_slope normalisiert | non-precision direction |
| nonprec_norm | non_prec normalisiert | non-precision coordinate |
| log1p(n_strips) | log(1 + N) | Clustergröße |
| sin(θ) | sin(theta_deg * π/180) | Winkel-Encoding |
| cos(θ) | cos(theta_deg * π/180) | Winkel-Encoding |
| muTPC_slope_norm | Slope aus (x,z)-Fit, normalisiert | gemessener Winkel aus Hits |
| q_asym | (q_back - q_front) / q_total | Ladungsasymmetrie (Vogel §5.4) |

n_global_feats: 3 → 7

### Normalization-Erweiterung
Neue Normierungsstatistiken nötig: `muTPC_slope_mean`, `muTPC_slope_std`.
sin/cos sind per Konstruktion in [-1,1] — keine Normierung nötig.
q_asym ist per Konstruktion in [-1,1] — keine Normierung nötig.

---

## Offene Fragen / TODO

- [ ] Multi-Winkel-Training: 29° + 15° neu trainieren, Leave-one-out auf 10°, 20°, 24°
- [ ] H4_GIF Tail-Problem: Ist es Hintergrund oder Normierungsproblem?
- [ ] H8_29° σ_core=166µm schlechter als H4_29° σ_core=113µm — Detektorunterschiede?
- [ ] 200ns-Datensätze: t=-inf in Zeitbranch, Zero-Shot strukturell nicht möglich ohne Re-Normierung
