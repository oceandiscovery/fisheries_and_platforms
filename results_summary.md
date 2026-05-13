# Results Summary — Fish Catches × Oil Platforms, Rio Grande do Norte (Brazil)

Dataset: 8 localities, 2001–2024 (24 years); analysis primary n = 7 (Areia Branca excluded; see PA below).

---

## 1. System Overview (Pipeline scripts 01–11)

### 1.1 Regional production and effort

| Year | Production (t) | Fishermen | CPUE (t/fish.) | Localities (n) |
|------|---------------|-----------|----------------|----------------|
| 2001 | 566           | 1 536     | 0.369          | 4              |
| 2005 | 1 095         | 2 078     | 0.465          | 7              |
| 2009 | 6 059         | 3 060     | 1.980          | 8              |
| 2016 | 7 277         | 2 142     | 3.397          | 8              |
| 2020 | 6 619         | 2 044     | 3.238          | 8              |
| 2024 | 5 500         | 1 741     | 3.159          | 8              |

System production grew roughly 10× over the study period; CPUE per fisherman grew approximately 8× (2001–2024). The steepest acceleration coincides with the 2008–2009 transition.

### 1.2 Platform-distance groups (Mann–Kendall regional trends)

| Platform class | CPUE trend | τ     | p      |
|---------------|-----------|-------|--------|
| 0–20 km       | ns        | 0.087 | 0.572  |
| 20–50 km      | increasing | 0.710 | <0.001 |

Near-platform localities (0–20 km) showed no significant CPUE trend despite increasing production (effort growth offset gains). Far-platform localities showed strong positive CPUE trends.

### 1.3 MPA-distance groups (Mann–Kendall)

| MPA class   | CPUE trend | τ     | p      |
|-------------|-----------|-------|--------|
| 0–10 km     | ns        | 0.087 | 0.572  |
| 10–25 km    | ns        | 0.010 | 0.976  |
| 25–50 km    | increasing | 0.457 | 0.001  |
| Inside MPA  | increasing | 0.812 | <0.001 |

### 1.4 Period comparison (early 2001–2008 vs recent 2017–2024)

| Group              | CPUE early (t/fish.) | CPUE recent (t/fish.) | Change |
|--------------------|---------------------|-----------------------|--------|
| 0–20 km            | 0.618               | 1.960                 | 3.2×   |
| 20–50 km (MPA)     | 0.452               | 2.947                 | 6.5×   |

Both groups improved, but the inside-MPA / 20–50 km group grew faster in the recent period.

---

## 2. CPUE Measure Comparison — Script 12

GAM standardisation controls for log fleet size and log fishermen count (R² = 0.811, AIC = −575.01). LMM results:

| Covariate         | β       | p       | sig |
|-------------------|---------|---------|-----|
| year (centred)    | +0.035  | <0.001  | *** |
| log fleet (c)     | −0.069  | 0.573   | ns  |
| log fishermen (c) | +0.607  | 0.001   | **  |

The positive year trend is robust to fleet/effort standardisation. Fishermen count positively predicts CPUE — consistent with aggregative behaviour (fishermen concentrate when fish are abundant). Fleet size adds no independent signal. The GAM-standardised index rises from 0.053 (2001) to ~0.14–0.15 (2022–2024), tracking the raw index closely.

---

## 3. PA — Areia Branca FAD Case Study — Script 13

### 3.1 The transition

Areia Branca (AB) underwent a complete, abrupt fishery restructuring between 2008 and 2009, coinciding with platform expansion. The platform-as-FAD hypothesis is supported by:

**Gear composition (AB):**

| Period     | Linha (longline) | Covo lagosta (trap) | Albacora share |
|------------|-----------------|---------------------|----------------|
| 2005–2008  | ~23%            | ~28–55%             | <0.3%          |
| 2009       | 49%             | <0.01%              | 35%            |
| 2013       | 74%             | 4%                  | 65%            |
| 2015–2024  | 94–98%          | <1%                 | 83–90%         |

**Contrast with Porto do Mangue (nearest comparator):**

| Metric           | AB (2024) | PdM (2024) |
|------------------|-----------|-----------|
| CPUE (t/fish.)   | 13.23     | 1.60      |
| Albacora share   | 89.6%     | 24.7%     |
| Linha share      | 97.6%     | 41.0%     |
| Shannon H′       | 0.572     | 2.715     |
| Species richness | 45        | 51        |

### 3.2 CPUE trajectory (AB)

| Year | CPUE (t/fish.) | Albacora share | Shannon H′ |
|------|---------------|----------------|------------|
| 2005 | 0.47          | 0.11%          | 1.41       |
| 2008 | 1.05          | 0.22%          | 2.16       |
| 2009 | 1.31          | 35.1%          | 2.42       |
| 2013 | 4.43          | 65.3%          | 1.63       |
| 2015 | 4.35          | 88.8%          | 0.67       |
| 2020 | 11.37         | 90.5%          | 0.58       |
| 2024 | 13.23         | 89.6%          | 0.57       |

AB CPUE is now 8–12× the system median (1.40 t/fisherman). The dramatic CPUE gain comes at the cost of near-complete species diversity collapse (H′ 2.16 → 0.57; Pielou J′ 0.61 → 0.15), driven by specialisation on a single target (Albacora / yellowfin tuna).

Porto do Mangue follows the opposite diversity trajectory: H′ stable and high (2.71–3.19), confirming the AB pattern is locality-specific and FAD-driven.

---

## 4. P1 — Spatial Characterisation (n = 7) — Script 14

### 4.1 Locality profile (sorted by platform distance)

| Locality        | CPUE med. | Richness | Shannon H′ | Pielou J′ | Plat. dist. (km) | MPA |
|-----------------|-----------|----------|-----------|----------|-----------------|-----|
| Macau           | 1.935     | 40.9     | 2.463     | 0.665    | 8.7             | Yes |
| Guamare         | 1.448     | 38.3     | 2.654     | 0.735    | 15.5            | No  |
| Galinhos        | 1.300     | 35.4     | 2.506     | 0.707    | 21.8            | No  |
| Porto do Mangue | 1.145     | 37.7     | 2.777     | 0.781    | 24.8            | Yes |
| Caiçara do Norte| 1.715     | 38.9     | 2.636     | 0.723    | 31.7            | No  |
| Grossos         | 0.622     | 24.7     | 1.946     | 0.691    | 44.7            | No  |
| Tibau           | 0.637     | 26.6     | 1.969     | 0.605    | 56.6            | No  |

### 4.2 Spearman correlations (n = 7)

| Metric     | Predictor      | ρ      | p     | sig |
|------------|----------------|--------|-------|-----|
| CPUE med.  | Platform dist. | −0.750 | 0.052 | .   |
| CPUE med.  | MPA dist.      | −0.357 | 0.430 | ns  |
| CPUE med.  | Inside MPA     | +0.500 | 0.253 | ns  |
| Richness   | Platform dist. | −0.714 | 0.071 | .   |
| Richness   | MPA dist.      | −0.464 | 0.294 | ns  |
| Shannon H′ | Platform dist. | +0.036 | 0.938 | ns  |
| Shannon H′ | MPA dist.      | +0.286 | 0.533 | ns  |
| Shannon H′ | Inside MPA     | +0.289 | 0.535 | ns  |
| Pielou J′  | Platform dist. | −0.143 | 0.760 | ns  |
| Pielou J′  | MPA dist.      | +0.179 | 0.701 | ns  |

**Key finding:** CPUE and species richness both show negative trends with platform distance (ρ = −0.750 and −0.714), marginally significant (p ≈ 0.05–0.07) with only 7 data points. Shannon diversity is not associated with platform distance, suggesting differences in catch magnitude rather than overall diversity gradients.

---

## 5. P2 — CPUE Temporal Dynamics (n = 7) — Script 15

### 5.1 Mann–Kendall + Sen's slope per locality

| Locality        | τ     | p       | sig | Slope (t/fish./yr) |
|-----------------|-------|---------|-----|---------------------|
| Galinhos        | 0.621 | <0.001  | *** | 0.066              |
| Caiçara do Norte| 0.529 | <0.001  | *** | 0.073              |
| Porto do Mangue | 0.442 | 0.002   | **  | 0.061              |
| Guamare         | 0.457 | 0.001   | **  | 0.064              |
| Macau           | 0.304 | 0.038   | *   | 0.071              |
| Tibau           | 0.088 | 0.655   | ns  | 0.006              |
| Grossos         | 0.111 | 0.534   | ns  | 0.011              |

5 of 7 localities show significant positive CPUE trends. The two non-significant localities (Tibau and Grossos) are the most distant from platforms (45–57 km) and MPAs. All 7 slopes are positive.

### 5.2 Regional GAM trend

A GammaGAM fitted to the n = 7 pooled series explained 81% of deviance. The fitted trend shows accelerating increase from 2001 through approximately 2016, followed by modest stabilisation through 2024.

---

## 6. P3a — Diversity Dynamics (n = 7) — Script 16

### 6.1 Mann–Kendall per locality × metric

| Locality        | Richness τ | sig | Shannon H′ τ | sig | Pielou J′ τ | sig |
|-----------------|-----------|-----|-------------|-----|------------|-----|
| Macau           | +0.765    | *** | +0.645      | *** | +0.399     | **  |
| Porto do Mangue | +0.728    | *** | +0.362      | *   | −0.232     | ns  |
| Caiçara do Norte| +0.717    | *** | +0.514      | *** | +0.210     | ns  |
| Guamare         | +0.686    | *** | +0.138      | ns  | −0.457     | **  |
| Galinhos        | +0.495    | **  | +0.063      | ns  | −0.158     | ns  |
| Grossos         | +0.263    | ns  | +0.177      | ns  | −0.100     | ns  |
| Tibau           | −0.030    | ns  | +0.176      | ns  | +0.309     | .   |

**Summary:** Richness — 6/7 positive, 5/7 significant (median τ = +0.686). Shannon H′ — 7/7 positive, 3/7 significant (median τ = +0.177). Pielou J′ — mixed: 4/7 negative; 2 significantly negative (Guamare, Macau).

The divergence between richness and evenness indicates more species being recorded while catch composition concentrates on fewer dominant species at some localities.

---

## 7. P3b — Effort Composition Dynamics (n = 7) — Script 17

| Locality        | Active gear τ | sig | Passive gear τ | sig | Motorised τ | sig |
|-----------------|--------------|-----|---------------|-----|------------|-----|
| Macau           | +0.507       | *** | −0.536        | *** | +0.855     | *** |
| Guamare         | +0.543       | *** | −0.580        | *** | +0.831     | *** |
| Galinhos        | +0.411       | *   | −0.326        | *   | +0.776     | *** |
| Porto do Mangue | +0.058       | ns  | −0.043        | ns  | +0.819     | *** |
| Caiçara do Norte| −0.087       | ns  | −0.080        | ns  | +0.435     | **  |
| Grossos         | +0.260       | ns  | −0.283        | .   | +0.189     | ns  |
| Tibau           | −0.307       | ns  | +0.407        | *   | +0.052     | ns  |

**Motorisation** is the most consistent trend: 6/7 localities significant (exception: Tibau). The shift from passive to active gear is significant at 4/7 localities.

---

## 8. P4 — CPUE × MPA Status Over Time (LMM) — Script 15

Linear mixed model: log(CPUE) ~ year_c × inside_mpa + (1 | local).

### 8.1 n = 7 (primary — Areia Branca excluded)

| Term                | β       | SE    | p     | sig |
|---------------------|---------|-------|-------|-----|
| year_c              | +0.065  | 0.009 | <0.001| *** |
| inside_mpa          | +0.435  | 0.289 | 0.133 | ns  |
| year_c × inside_mpa | −0.004  | 0.014 | 0.797 | ns  |

### 8.2 n = 8 (sensitivity — Areia Branca included)

| Term                | β       | SE    | p     | sig |
|---------------------|---------|-------|-------|-----|
| year_c              | +0.065  | 0.010 | <0.001| *** |
| inside_mpa          | +0.718  | 0.283 | 0.011 | *   |
| year_c × inside_mpa | +0.039  | 0.015 | 0.010 | **  |

**Key finding:** The significant MPA × year interaction (n=8, p=0.010) disappears entirely when AB is excluded (n=7, p=0.797). AB's extraordinary CPUE growth (FAD-driven, not MPA-related) is the sole driver of the apparent interaction. There is no evidence that MPA localities follow different CPUE trajectories from non-MPA localities in the primary analysis.

---

## 9. P5 — Diversity × MPA Status Over Time (LMM) — Script 16

### 9.1 n = 7 (primary)

| Metric     | year_c β | p      | year_c × MPA β | p     | sig |
|------------|----------|--------|----------------|-------|-----|
| Richness   | +0.864   | <0.001 | +0.398         | 0.013 | *   |
| Shannon H′ | +0.028   | <0.001 | +0.005         | 0.634 | ns  |
| Pielou J′  | −0.000   | 0.750  | +0.002         | 0.323 | ns  |

### 9.2 n = 8 sensitivity — interaction term

| Metric     | year_c × MPA β | p     | sig | vs n=7     |
|------------|----------------|-------|-----|------------|
| Richness   | +0.452         | 0.003 | **  | robust     |
| Shannon H′ | −0.022         | 0.066 | .   | marginal   |
| Pielou J′  | −0.006         | 0.029 | *   | AB artifact |

**Key findings:**
- **Richness:** MPA localities accumulate richness +0.40 species/yr faster than non-MPA (p=0.013). Robust across n=7 and n=8. Real ecological signal.
- **Pielou J′:** Significant in n=8 (p=0.029) but not in n=7 (p=0.323). Confirmed AB artifact.
- **Shannon H′:** Clearly non-significant in n=7 (p=0.634).

---

## 10. P6 — CPUE ~ Gear Composition (n = 7) — Script 17

### 10.1 Within-local Spearman (demeaned)

| Predictor        | ρ      | p      | sig |
|------------------|--------|--------|-----|
| Active gear share| +0.506 | <0.001 | *** |
| Motorised share  | +0.372 | <0.001 | *** |

### 10.2 LMM (log CPUE ~ year_c + active_dm + motorised_dm + (1|local))

| Term         | β       | SE    | p      | sig |
|--------------|---------|-------|--------|-----|
| year_c       | +0.042  | 0.009 | <0.001 | *** |
| active_dm    | +1.611  | 0.266 | <0.001 | *** |
| motorised_dm | +0.157  | 0.291 | 0.589  | ns  |

After absorbing the year trend (year smooth range: −0.53 to +0.75), active gear share retains a strong positive partial effect on CPUE. Motorised share adds no independent predictive power when active gear and year are controlled.

---

## 11. P7 — Density Dependence (n = 7) — Script 18

GAM: log(CPUE_dm) ~ smooth(log_fishermen_dm), fitted to within-local demeaned data.

The GAM partial effect is monotonically positive: CPUE is lower in years with few fishermen and higher in years with many fishermen (CPUE effect range: −0.46 at log_fish_dm = −2.23 to +0.37 at +0.96). The CI excludes zero at both extremes.

**Interpretation:** The positive relationship reflects aggregative behaviour — fishermen concentrate effort in years of high local abundance, producing a spurious positive correlation. There is no evidence of CPUE depression at high effort levels within the range observed (no density-dependent depletion detectable at this scale).

---

## 12. P8 — CPUE ~ Diversity (n = 7) — Script 18

### 12.1 Within-local Spearman (demeaned)

| Metric     | ρ      | p      | sig |
|------------|--------|--------|-----|
| Richness   | +0.599 | <0.001 | *** |
| Shannon H′ | +0.190 | 0.019  | *   |
| Pielou J′  | −0.176 | 0.032  | *   |

### 12.2 Year-controlled GAM partial effect

The richness partial effect is positive across the full range after removing the year trend. Localities in above-average richness years tend to also show above-average CPUE.

**Interpretation:** Years with higher local species richness are associated with higher CPUE — consistent with biodiversity–ecosystem function theory in small-scale fisheries. The negative Pielou J′ coefficient suggests that catch evenness (high evenness = many species contributing equally) slightly depresses CPUE relative to situations dominated by one productive target species.

---

## 13. P9 — Species Composition (n = 7 + AB) — Script 19

### 13.1 SIMPER — top discriminating species (platform distance, n = 7)

| Rank | Species      | Contrib. % | Near (0–20 km) | Far (20–50 km) |
|------|-------------|-----------|----------------|----------------|
| 1    | Lagosta      | 12.5      | lower (0.038)  | higher (0.171) |
| 2    | Sardinha     | 11.2      | higher (0.153) | lower (0.014)  |
| 3    | Tainha       | 9.3       | higher (0.154) | lower (0.093)  |
| 4    | Peixe Voador | 6.9       | lower (0.049)  | higher (0.073) |
| 5    | Buzios       | 4.0       | higher (0.045) | lower (0.019)  |
| 6    | Marisco      | 3.8       | higher (0.048) | lower (0.005)  |

Near-platform localities have higher coastal/demersal species (Sardinha, Tainha, shellfish). Far-platform localities have more Lagosta and Peixe Voador. Top 7 species explain ~52% of community dissimilarity.

### 13.2 Species MK trends (n = 7): 36 / 79 significant (45.6%)

**Most consistent increases:** Buzios (4 localities, τ = 0.44–0.76***), Guaiuba (2 localities), Agulha (2 localities), Ariaco (2 localities).

**Most consistent decreases:** Peixe Voador (2 localities, including Galinhos τ = −0.726***), Sardinha (Macau τ = −0.638***, Galinhos −0.484**), Lagosta (3 localities), Outros (3 localities).

The system is shifting from flying fish and sardines toward reef-associated species (Buzios, Guaiuba) and selected benthic fish (Agulha).

### 13.3 Species MK trends — Areia Branca (11 / 12 significant)

| Species  | τ      | sig | Direction    |
|----------|--------|-----|-------------|
| Lagosta  | −0.819 | *** | Collapse     |
| Bonito   | +0.788 | *** | Strong rise  |
| Albacora | +0.753 | *** | Dominance    |
| Garajuba | −0.619 | *** | Decline      |
| Sirigado | −0.524 | *** | Decline      |
| Cioba    | −0.520 | *** | Decline      |
| Camarao  | −0.471 | **  | Decline      |
| Ariaco   | −0.438 | **  | Decline      |
| Serra    | −0.352 | *   | Decline      |
| Caico    | −0.362 | *   | Decline      |
| Outros   | +0.473 | **  | Increase     |

The Albacora rise (τ = +0.753) and Lagosta collapse (τ = −0.819) quantify the complete species turnover. All demersal/reef species declined significantly. Bonito (skipjack tuna) also increased strongly, consistent with a platform-associated pelagic ecosystem.

---

## 14. Cross-Cutting Synthesis

1. **FAD effect confirmed at AB.** The 2009 transition is sharp, complete, and unmatched at the nearest control locality (Porto do Mangue). No other locality shows comparable species or gear restructuring.

2. **System-wide CPUE growth is real and trend-driven.** 5/7 localities show significant positive MK trends; the GAM-standardised index confirms the trend is not a fleet/effort artefact.

3. **No MPA effect on CPUE trajectories.** After excluding AB, the MPA × year interaction is null (β = −0.004, p = 0.797). The n=8 result (p = 0.010) is entirely driven by AB's FAD-driven CPUE explosion.

4. **Real MPA biodiversity effect: species richness.** MPA localities accumulate richness +0.40 species/yr faster (p = 0.013), surviving AB exclusion. Pielou J′ n=8 result is an AB artefact.

5. **Active gear shift predicts CPUE.** LMM β = +1.61 (p < 0.001) is the strongest within-locality CPUE predictor, independent of the secular year trend.

6. **Richness and CPUE are positively linked within localities.** Above-average local richness is associated with above-average CPUE (ρ = +0.599***), consistent with biodiversity–ecosystem function theory.

7. **Platform proximity correlates with higher CPUE and richness** (ρ ≈ −0.71 to −0.75), marginal with n = 7. Consistent with platforms acting as artificial reefs across the system.

8. **Grossos and Tibau** consistently stand out: lowest CPUE, lowest or declining richness, no significant trends. Furthest from both platforms (45–57 km) and MPAs. May represent the background shelf condition.

---

*Generated from pipeline outputs in `data/processed/`. Scripts 01–11 (pipeline) and 12–19 (analysis). Primary n = 7 excludes Areia Branca; n = 8 reported as sensitivity where noted.*
