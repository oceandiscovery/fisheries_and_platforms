# Research Questions — Fish × Platforms Brazil (v2)
### Revisión post-diagnóstico · Mayo 2026

---

## Marco analítico

El análisis de la composición de especies reveló que **Areia Branca (AB) es un caso singular** que no es
representativo de las dinámicas generales del sistema. AB experimentó una transición pesquera completa
(2008–2009) desde una pesquería multiespecífica dominada por Lagosta hacia una pesquería de Albacora
(*Thunnus albacares*) casi exclusiva mediante palangre (*linha*), pasando de <0.2 % a 85–91 % de
participación de Albacora en la captura. Esta transición infla artificialmente los resultados de las
preguntas P4 y P5 originales, que detectaban una "señal MPA" que en realidad era la firma temporal de
la especialización en Albacora en una sola localidad.

Las preguntas se organizan en dos bloques:

- **Bloque A** — Areia Branca como caso de estudio de interacción plataforma-pesquería (hipótesis FAD)
- **Bloque B** — Las 7 localidades restantes: señal ecológica general sobre efectos de plataformas y MPAs

---

## Bloque A — Areia Branca: evidencia para la hipótesis FAD

### PA — ¿Actuaron las plataformas offshore como dispositivos de agregación (FAD) para grandes
pelágicos en Areia Branca?

**Contexto:** Los grandes pelágicos —especialmente el atún aleta amarilla (*Albacora*)— tienen una
tendencia documentada a agregarse alrededor de estructuras artificiales sumergidas. Areia Branca está
a 31 km de la plataforma más cercana (clase 20–50 km) y es la única localidad del sistema que muestra
una especialización masiva en Albacora. Porto do Mangue, con exposición y distancia similares, no
experimentó la misma transición.

**Sub-preguntas:**

- PA.1 — ¿Es la cronología de la transición (2008–2009) consistente con la expansión de operaciones
  de plataformas en el área?
- PA.2 — ¿La reorganización del esfuerzo (abandono de covo lagosta, adopción masiva de *linha*)
  es coherente con el redireccionamiento hacia especies pelágicas offshore?
- PA.3 — ¿Qué diferencia a Areia Branca de Porto do Mangue —localidad con clase de exposición
  idéntica— que no realizó la misma transición? (distancia exacta, capacidad de flota,
  infraestructura de desembarco, acceso a mercado)
- PA.4 — ¿El patrón de diversidad en AB (riqueza alta + equitatividad colapsada) es consistente
  con el bycatch incidental de una pesquería de palangre oceánico orientada a un único target?

**Nivel de identificación:** Descriptivo-mecanístico; no experimental. La inferencia causal es
indirecta (cronología, especificidad espacial, ecología de la especie). No requiere modelo estadístico
formal; requiere descripción detallada y comparación con Porto do Mangue.

**Outputs clave:** Serie temporal de composición de especies y artes en AB; comparación AB vs.
Porto do Mangue; diagrama de transición pesquera.

---

## Bloque B — Las 7 localidades: señal ecológica general

> Todas las preguntas de este bloque se aplican al conjunto de 7 localidades **excluyendo Areia Branca**,
> salvo indicación expresa. Donde el análisis completo (n=8) se reporta, se hace explícitamente como
> análisis de sensibilidad.

---

### P1 — ¿Cómo varía la CPUE y la diversidad de especies entre localidades en relación con
la distancia a plataformas y a MPAs?

**Tipo:** Caracterización espacial entre-localidades (estimador between, n=7).

**Identificabilidad:** La distancia a plataformas y el estatus MPA son constantes por localidad
→ solo comparables entre localidades. Con n=7 la potencia es baja; se reportan correlaciones de
Spearman y gráficos de dispersión etiquetados, sin inferencia formal sobre significancia.

**Métricas:** CPUE mediana (t/pescador), riqueza media, Shannon H' medio, por localidad.

**Pregunta operativa:** ¿Existe un gradiente negativo de CPUE o diversidad con la distancia a
plataformas? ¿Las localidades dentro de MPAs tienen mayor CPUE o diversidad media?

---

### P2 — ¿Cuáles son las tendencias temporales de CPUE dentro de las localidades y en
qué medida son consistentes entre clases de exposición?

**Tipo:** Within-locality Mann-Kendall + GAM global; n=7 localidades, ~15–24 años c/u.

**Identificabilidad:** Las tendencias within-locality son bien identificadas. La comparación entre
clases de exposición es descriptiva (n demasiado pequeño para inferencia formal between-class).

**Métricas:** τ de Kendall, pendiente de Sen (t/pescador/año), tendencia GAM regional.

**Pregunta operativa:** ¿Predominan las tendencias positivas o negativas? ¿Son más fuertes en
localidades cercanas a plataformas o dentro de MPAs?

---

### P3 — ¿Cuáles son las tendencias temporales en diversidad de especies y composición
del esfuerzo (artes, embarcaciones) dentro de las localidades?

**Tipo:** Within-locality Mann-Kendall para riqueza, H', J' y shares de artes/propulsión.

**Sub-preguntas:**

- P3a — ¿Están aumentando la riqueza y la diversidad a lo largo del tiempo dentro de las
  localidades? ¿Es la tendencia más pronunciada en localidades con mayor exposición a plataformas
  o dentro de MPAs?
- P3b — ¿Existe una transición regional hacia embarcaciones motorizadas y artes activas?
  ¿Es este cambio universal o diferencial por clase de exposición?

**Identificabilidad:** Bien identificada within-locality. La comparación entre clases es descriptiva.

---

### P4 — ¿Difieren las trayectorias de CPUE a lo largo del tiempo entre localidades
con diferente exposición a plataformas y MPAs?

**Tipo:** LMM con interacción `año × exposición` + GAMs estratificados; n=7.

**Identificabilidad:** La interacción `año_c × inside_mpa` es identificable en presencia de efectos
aleatorios por localidad porque el término de interacción varía dentro de cada localidad a lo largo
del tiempo. El efecto nivel (inside_mpa main effect) no es identificable con RE locales.

**Análisis de sensibilidad:** Se reporta también el resultado con n=8 (incluyendo AB) para documentar
el tamaño del sesgo que introduce AB.

**Pregunta operativa:** ¿Hay evidencia de que CPUE creció más rápido en localidades dentro de MPAs
o más cercanas a plataformas, una vez excluida la singularidad de Areia Branca?

---

### P5 — ¿Difieren las trayectorias de diversidad (riqueza, Shannon H', Pielou J')
entre localidades con diferente exposición, excluyendo Areia Branca?

**Tipo:** LMM con interacción `año × inside_mpa` para cada métrica de diversidad; n=7.

**Identificabilidad:** Misma lógica que P4. La interacción temporal es identificable; el efecto
nivel no lo es.

**Análisis de sensibilidad:** Resultado con n=8 para cuantificar la influencia de AB en la
interacción richness × inside_mpa (donde la señal puede ser parcialmente real) y en la interacción
Pielou × inside_mpa (donde la señal es espuria con alta probabilidad).

**Pregunta operativa:** ¿El aumento de riqueza es significativamente más pronunciado within
localidades de MPA? ¿O es un patrón regional uniforme?

---

### P6 — Within-locality: ¿está la CPUE asociada con la composición del esfuerzo
pesquero (artes, propulsión), controlando por la tendencia temporal no lineal?

**Tipo:** Within-locality Spearman (demeaned) + LMM + LinearGAM con `s(arte_dm) + s(motor_dm) +
s(año_c)`.

**Justificación del control por año:** La motorización es una tendencia temporal universal (P3b).
Sin controlar por `s(año_c)`, el efecto del share motorizado podría reflejar confusión con la
tendencia temporal compartida. El GAM con año explícito separa el efecto composicional puro del
trend.

**Pregunta operativa:** ¿El efecto positivo de artes activas sobre CPUE (β=+1.07** en el modelo
sin control temporal) persiste después de controlar por `s(año)`? ¿El efecto negativo de
motorización (β=−0.74*) es real o artefacto temporal?

---

### P7 — Within-locality: ¿existe una relación densidad-dependiente entre nivel de
esfuerzo y CPUE?

**Tipo:** LinearGAM `s(log_pescadores_dm)` sobre datos demeaned within-locality.

**Identificabilidad:** Bien identificada; la variación within-local en número de pescadores es
independiente de la exposición espacial.

**Pregunta operativa:** ¿Años con más pescadores activos tienen menor CPUE (densidad-dependencia
negativa), o mayor (efecto escala/tecnología)?

---

### P8 — Within-locality: ¿está la CPUE asociada con la diversidad de especies,
controlando por la tendencia temporal no lineal?

**Tipo:** Within-locality Spearman (demeaned) + LinearGAM con `s(diversidad_dm) + s(año_c)`.

**Justificación del control por año:** La correlación negativa entre Pielou y CPUE (ρ=−0.34**
en el modelo sin control) puede ser artefacto de la tendencia temporal de AB, en la cual CPUE y
Pielou se mueven simultáneamente en respuesta a la expansión de Albacora. Controlando por año se
aísla la asociación pura within-local.

**Pregunta operativa:** ¿Más riqueza within-local se asocia con mayor CPUE (interpretación:
años más productivos detectan más especies), independientemente del trend temporal? ¿La
correlación negativa con Pielou persiste o desaparece al controlar por año?

---

### P9 — ¿Qué especies explican las diferencias en composición de la captura entre
localidades, y cómo ha evolucionado la dominancia de especies a lo largo del tiempo?

**Tipo:** SIMPER por pares de grupos + Mann-Kendall de abundancia relativa por localidad×especie.

**Sub-preguntas:**

- P9a — ¿Qué especies contribuyen más a la disimilitud Bray-Curtis entre localidades de diferente
  clase de exposición a plataformas y MPAs?
- P9b — ¿Hay especies con tendencias temporales significativas dentro de localidades específicas?
  ¿Son estas tendencias consistentes con los efectos de exposición o son idiosincrásicas?

**Nota:** El SIMPER para el contraste "inside vs. outside MPA" está dominado por Albacora
(contribución del 13.7 %) debido exclusivamente a AB. La interpretación debe separar el resultado
AB del resultado para las otras localidades inside MPA (Porto do Mangue, Macau).

---

## Resumen de la estructura analítica

| Pregunta | n | Análisis principal | Identifica |
|---|---|---|---|
| **PA** | AB vs. Porto do Mangue | Descriptivo-mecanístico | FAD hypothesis (indirecta) |
| **P1** | 7 | Spearman between-estimator | Gradiente espacial CPUE/diversidad |
| **P2** | 7 | Mann-Kendall + GAM | Tendencias CPUE within |
| **P3** | 7 | Mann-Kendall (diversidad + esfuerzo) | Tendencias diversidad y motorización |
| **P4** | 7 (+8 sensitivity) | LMM `año × MPA` + GAM | Interacción tendencia × exposición |
| **P5** | 7 (+8 sensitivity) | LMM `año × MPA` + GAM | Interacción tendencia diversidad × exposición |
| **P6** | 7 | GAM `s(arte) + s(motor) + s(año)` | Efecto composición gear sobre CPUE |
| **P7** | 7 | LinearGAM `s(esfuerzo_dm)` | Densidad-dependencia within-local |
| **P8** | 7 | GAM `s(diversidad) + s(año)` | Asociación diversidad-CPUE within-local |
| **P9** | 7 (+AB separado) | SIMPER + MK por especie | Composición y tendencias de especies |

---

## Nota metodológica: tratamiento de Areia Branca

Areia Branca **no se excluye del dataset**; se trata de forma diferenciada:

1. Para PA: se analiza en detalle como caso de estudio.
2. Para P1–P9: el análisis principal usa n=7; el resultado con n=8 se reporta como análisis
   de sensibilidad para cuantificar la influencia de AB sobre cada estimado.
3. En figuras: AB se identifica visualmente (símbolo distinto, anotación) cuando se incluye
   en representaciones de las 8 localidades.

Esta estructura permite mantener la transparencia sobre el caso singular sin descartar sus datos,
y separar la hipótesis FAD de las preguntas ecológicas generales.

---

*Documento generado: 2026-05-13 | Proyecto: Fish × Platforms — Rio Grande do Norte, Brazil*
