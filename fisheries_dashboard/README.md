Scientific Summary of the Project

⸻

Context and Rationale

The northern coast of Rio Grande do Norte state, Brazil, is a setting in which two highly relevant economic activities directly overlap: small-scale artisanal fisheries, on which many coastal communities depend as a primary source of income, employment, and food security, and offshore oil and gas extraction, whose infrastructure has progressively expanded along this same coastline over the last two decades. This coexistence has generated a persistent tension between traditional uses of marine space and new forms of industrial occupation, including platforms, cables, associated traffic, and operational restriction areas.

In this context, the establishment of the Programa de Monitoramento do Desembarque Pesqueiro (PMDP) represented a strategic effort to document, in a systematic way, the dynamics of artisanal landings before, during, and after the consolidation of offshore infrastructure. The value of the PMDP lies not only in the temporal continuity of its records, but also in its ability to provide an exceptional empirical basis for assessing whether the spatial footprint of the offshore oil industry leaves detectable signals in fishery production, species diversity, and the structure of landed assemblages.

Beyond sectoral coexistence, the issue has a broader ecological and territorial dimension. Offshore platforms are not merely extractive structures: they also introduce permanent hard substrate into a marine environment where such substrate may be relatively scarce, potentially affecting fish aggregation, local resource availability, and fishing accessibility. At the same time, this coastal-industrial seascape overlaps with marine protected areas and spaces of high ecological and social value, making it necessary to evaluate these patterns not in isolation, but within a more complex spatial context.

⸻

Central Research Question

To what extent is the spatial proximity of landing points to offshore oil platforms associated with changes in production, species diversity, and catch composition in artisanal fisheries along the northern coast of Rio Grande do Norte, while also considering the spatial context of marine protected areas?

⸻

Conceptual Framework and Hypotheses

The question is grounded in a widely documented, though still debated, ecological mechanism: submerged artificial structures, including platform bases, may function as artificial reefs capable of aggregating biomass and altering the local distribution of fish and invertebrates. This effect may translate into changes in catchability, species richness, and the taxonomic composition of catches. However, its consequences are not necessarily unambiguous, since the aggregation of organisms does not always imply an actual increase in ecological production, and its effects may depend on the fishing and spatial context in which it occurs.

H1 — Positive aggregation effect: Localities closer to platforms show higher production per trip, greater species richness, and higher diversity indices, consistent with a possible aggregation effect associated with offshore infrastructure.

H2 — Assemblage turnover: Proximity to platforms is associated with changes in the taxonomic composition of catches, differentially favoring certain species or ecological groups and generating a detectable compositional gradient among localities.

H3 — Spatial exposure gradient: There is a continuous spatial gradient of platform exposure, expressed through distances, local platform density, and proximity-weighted metrics, that explains a meaningful fraction of the variation observed in production, diversity, and catch composition.

H4 — Modulation by conservation context: The relationship between platform exposure and fishery responses may vary according to the spatial context of protected areas, such that the proximity of landing points to protected areas, or their partial inclusion within them, contributes to shaping the observed patterns.

⸻

Study Area

The study focuses on five coastal localities along the coast of Rio Grande do Norte: Areia Branca, Caiçara do Norte, Guamaré, Macau, and Porto do Mangue. These localities account for an important share of the PMDP fishery monitoring effort and are located along a coastal strip closely linked to the Potiguar Basin, the state’s main oil-producing region. The study system is therefore a marine-coastal setting in which artisanal fisheries, offshore infrastructure, and conservation-relevant areas converge.

⸻

Specific Objectives

Objective	Scripts
Clean and canonicalize landing, fleet, gear, species, and production value records	01 → 03
Resolve the spatial context of each landing point: municipality, nearest platforms, protected areas	04
Build locality-level exposure metrics: mean/minimum distance to platforms, number of platforms within 10–50 km buffers, inverse-distance sums, and distance to each protected area	04 → 05
Calculate diversity matrices (H’, J’, S) and species composition matrices by locality-year	06
Screen associations between exposure and response variables; assess predictor collinearity	07
Fit penalized GAMs and reference OLS models to quantify non-linear responses of production and diversity along the exposure gradient	08
Diagnose GAM stability and identify influential localities or years	09
Ordinate assemblages using PCoA (Bray-Curtis and Euclidean/Hellinger) and NMDS; test compositional differences among exposure groups with PERMANOVA and PERMDISP	10
Synthesize the primary compositional gradient and identify the species driving turnover along the exposure gradient	11

⸻

Analytical Unit and Data Structure

The fundamental analytical unit is the locality-year. Each of the five focal localities contributes one observation per year within the available monitoring period, yielding a small panel dataset with roughly one hundred observations in total. The main spatial predictors are derived from the fixed geometry of landing points, platform locations, and the protected-area context, and therefore remain essentially constant at the locality level. By contrast, response variables such as production, diversity, and catch composition vary from year to year.

This design provides a suitable basis for exploring persistent spatial associations, but it also imposes important inferential limitations. Given the small number of spatially independent units, the ability to disentangle purely spatial, temporal, and structural effects is necessarily limited. For this reason, the analysis adopts parsimonious models and a cautious interpretation, especially with regard to causality and generalization.

⸻

Key Variables

Spatial exposure predictors:

* Distance from each landing point to the nearest platform
* Aggregation of those distances at the locality level through statistics such as mean, median, or minimum
* Number of platforms within 10, 20, 30, and 50 km buffers
* Sum of inverse distances as a proximity-weighted exposure index
* Distance to the APA Dunas do Rosado and the RDS Ponta do Tubarão
* Proportion of each locality’s landing points located within, or relatively close to, each protected area

Response variables:

* Total production (tons), production per trip, and production per fisher
* Species richness (S), Shannon diversity (H’), and Pielou evenness (J’)
* Relative composition matrices and Hellinger-transformed composition matrices
* Production value per ton

⸻

Statistical Approach

The project combines three complementary analytical levels.

1. Univariate analysis
Associations between spatial predictors and response variables are evaluated using Spearman correlations and penalized generalized additive models, compared against reference ordinary least squares models. This approach allows the detection of both monotonic associations and non-linear responses along the spatial gradient. Multiple testing is controlled using Benjamini-Hochberg FDR correction.

2. Robustness diagnostics
Model stability and the influence of particular observations are examined, with special attention to whether the detected patterns depend disproportionately on specific localities or years. This step helps identify whether a substantial part of the observed signal is driven by a limited number of influential cases.

3. Multivariate analysis
Catch composition is analyzed using ordinations based on Bray-Curtis dissimilarity and Euclidean distance on Hellinger-transformed data, complemented by NMDS. PERMANOVA is then used to test for compositional differences among exposure groups, and PERMDISP is used to assess homogeneity of multivariate dispersion among groups. This latter step is essential, since a significant PERMANOVA can only be interpreted more strictly as a difference in group location when it is not accompanied by marked heterogeneity in within-group dispersion.

⸻

Expected Contribution

The project seeks to determine whether the spatial footprint of offshore oil infrastructure is detectably associated with the ecological and productive structure of artisanal fisheries along the northern coast of Rio Grande do Norte. More specifically, it aims to establish whether the gradient of platform exposure leaves a consistent signal in catch magnitude, landed species diversity, and assemblage composition, while also incorporating the spatial context of marine protected areas as a complementary element of interpretation.

The expected contribution is twofold. From a scientific perspective, the project provides spatially explicit evidence on how industrial marine infrastructure may be associated with change in small-scale coastal fishery systems. From an applied perspective, the results may inform decisions on exclusion-zone design, cumulative impact assessment, compensation for fishing communities affected by offshore industry, and the articulation between offshore infrastructure and marine spatial planning in settings where artisanal fisheries remain socially and economically important.