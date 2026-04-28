Resumen científico del proyecto

⸻

Contexto y motivación

El litoral norte del estado de Rio Grande do Norte (Brasil) constituye un espacio de contacto directo entre dos actividades económicas de gran relevancia regional: la pesca artesanal de pequeña escala, de la que dependen numerosas comunidades costeras como fuente principal de ingreso, empleo y seguridad alimentaria, y la explotación de petróleo y gas offshore, cuya infraestructura se ha expandido progresivamente frente a este mismo litoral durante las últimas dos décadas. Esta coexistencia ha generado una tensión persistente entre usos tradicionales del espacio marino y nuevas formas de ocupación industrial, incluyendo plataformas, cables, tráfico asociado y áreas de restricción operativa.

En este contexto, la instalación del Programa de Monitoramento do Desembarque Pesqueiro (PMDP) representó una respuesta estratégica para documentar, de manera sistemática, la dinámica de los desembarques artesanales antes, durante y después de la consolidación de esta infraestructura offshore. El valor del PMDP reside no solo en la continuidad temporal de sus registros, sino también en su capacidad para ofrecer una base empírica excepcional con la que evaluar si la huella espacial de la industria petrolera deja señales detectables en la producción pesquera, en la diversidad de especies capturadas y en la estructura de los ensamblajes desembarcados.

Más allá de la coexistencia sectorial, el problema tiene una dimensión ecológica y territorial más amplia. Las plataformas offshore no son únicamente infraestructuras extractivas: también introducen estructuras duras permanentes en un medio marino donde este tipo de sustrato puede ser escaso, generando efectos potenciales sobre la agregación de peces, la disponibilidad local de recursos y la accesibilidad pesquera. Al mismo tiempo, este paisaje costero-industrial se solapa con áreas marinas protegidas y con espacios de alto valor ecológico y social, lo que hace necesario evaluar estos patrones no de forma aislada, sino dentro de un contexto espacial más complejo.

⸻

Pregunta de investigación central

¿En qué medida la proximidad espacial de los puntos de desembarque a las plataformas petrolíferas offshore se asocia con cambios en la producción, la diversidad de especies y la composición de las capturas en las pesquerías artesanales del litoral norte de Rio Grande do Norte, considerando además el contexto espacial de las áreas marinas protegidas?

⸻

Marco conceptual e hipótesis

La pregunta se apoya en un mecanismo ecológico ampliamente documentado pero aún sujeto a debate: las estructuras artificiales sumergidas, incluidas las bases de plataformas, pueden funcionar como arrecifes artificiales capaces de agregar biomasa y modificar la distribución local de peces e invertebrados. Este efecto puede traducirse en cambios en la capturabilidad, en la riqueza de especies y en la composición taxonómica de las capturas. Sin embargo, sus consecuencias no son necesariamente unívocas, ya que la agregación de organismos no siempre implica incremento real de producción ecológica, y sus efectos pueden depender del contexto pesquero y espacial en el que se insertan.

H1 — Efecto de agregación positivo: Las localidades más próximas a plataformas presentan mayor producción por viaje, mayor riqueza de especies y mayores índices de diversidad, en concordancia con un posible efecto de agregación asociado a la infraestructura offshore.

H2 — Recambio composicional del ensamblaje: La proximidad a plataformas se asocia con cambios en la composición taxonómica de las capturas, favoreciendo diferencialmente determinadas especies o grupos ecológicos y generando un gradiente composicional detectable entre localidades.

H3 — Gradiente espacial de exposición: Existe un gradiente espacial continuo de exposición a plataformas, expresado mediante distancias, densidad local de plataformas y métricas ponderadas por proximidad, que explica una fracción relevante de la variación observada en producción, diversidad y composición de capturas.

H4 — Modulación por contexto de conservación: La relación entre exposición a plataformas y respuestas pesqueras puede variar según el contexto espacial de las áreas protegidas, de modo que la proximidad o inclusión parcial de puntos de desembarque en áreas protegidas contribuya a modular los patrones observados.

⸻

Área de estudio

El estudio se centra en cinco localidades costeras del litoral de Rio Grande do Norte: Areia Branca, Caiçara do Norte, Guamaré, Macau y Porto do Mangue. Estas localidades concentran una parte importante del monitoreo pesquero del PMDP y se ubican en una franja costera estrechamente vinculada a la cuenca Potiguar, principal región productora de petróleo del estado. Se trata, por tanto, de un sistema marino-costero donde convergen actividad pesquera artesanal, infraestructura offshore y áreas de interés para la conservación.

⸻

Objetivos específicos

Objetivo	Scripts
Limpiar y canonicalizar los registros de desembarque, flota, artes, especies y valor de producción	01 → 03
Resolver el contexto espacial de cada punto de desembarque: municipio, plataformas más cercanas, áreas protegidas	04
Construir métricas de exposición por localidad: distancia media/mínima a plataformas, número de plataformas en radios de 10–50 km, suma de distancias inversas, distancia a cada área protegida	04 → 05
Calcular matrices de diversidad (H’, J’, S) y matrices de composición de especies por localidad-año	06
Cribar asociaciones entre exposición y respuestas; evaluar colinealidad entre predictores	07
Ajustar GAMs penalizados y OLS de referencia para cuantificar la respuesta no lineal de producción y diversidad al gradiente de exposición	08
Diagnosticar estabilidad de los GAMs e identificar localidades o años influyentes	09
Ordenar los ensamblajes mediante PCoA (Bray-Curtis y Euclidiana/Hellinger) y NMDS; probar diferencias composicionales entre grupos de exposición con PERMANOVA y PERMDISP	10
Sintetizar el gradiente composicional primario e identificar las especies que definen el recambio a lo largo del gradiente de exposición	11

⸻

Unidad analítica y estructura de datos

La unidad analítica fundamental es la localidad-año. Cada una de las cinco localidades focales aporta una observación por año dentro del periodo de seguimiento disponible, configurando una serie panel de tamaño reducido, con aproximadamente un centenar de observaciones en total. Los predictores espaciales principales se derivan de la geometría fija de los puntos de desembarque, de la localización de las plataformas y del contexto de áreas protegidas, por lo que permanecen esencialmente constantes a escala de localidad. En cambio, las variables de respuesta, tales como producción, diversidad y composición de capturas, varían interanualmente.

Este diseño ofrece una base adecuada para explorar asociaciones espaciales persistentes, pero también impone limitaciones inferenciales importantes. Dado el bajo número de unidades espaciales independientes, la capacidad para separar con claridad efectos puramente espaciales, temporales y estructurales es necesariamente limitada. Por ello, el análisis adopta modelos parsimoniosos y una interpretación prudente, especialmente en lo relativo a causalidad y generalización.

⸻

Variables clave

Predictores espaciales de exposición:

* Distancia desde cada punto de desembarque a la plataforma más cercana
* Agregación de esas distancias a escala de localidad mediante estadísticas como media, mediana o mínimo
* Número de plataformas dentro de radios de 10, 20, 30 y 50 km
* Suma de distancias inversas como índice de exposición ponderado por proximidad
* Distancia a la APA Dunas do Rosado y a la RDS Ponta do Tubarão
* Proporción de puntos de desembarque de cada localidad dentro de cada área protegida o en su proximidad relativa

Variables de respuesta:

* Producción total (toneladas), producción por viaje y producción por pescador
* Riqueza de especies (S), diversidad de Shannon (H’) y equitatividad de Pielou (J’)
* Matrices de composición relativa y transformada de Hellinger de las capturas
* Valor de producción por tonelada

⸻

Enfoque estadístico

El proyecto articula tres niveles analíticos complementarios.

1. Análisis univariado
Se evalúan asociaciones entre predictores espaciales y variables de respuesta mediante correlaciones de Spearman y modelos aditivos generalizados penalizados, comparados con modelos lineales ordinarios de referencia. Este enfoque permite detectar tanto relaciones monotónicas como respuestas no lineales a lo largo del gradiente espacial. La significancia de múltiples asociaciones se controla mediante corrección FDR de Benjamini-Hochberg.

2. Diagnóstico de robustez
Se examina la estabilidad de los modelos y la influencia de observaciones particulares, con especial atención a la posible dependencia de los patrones respecto a localidades o años concretos. Este paso permite identificar si una parte importante de la señal detectada proviene de unos pocos casos especialmente influyentes.

3. Análisis multivariado
La composición de capturas se analiza mediante ordenaciones basadas en Bray-Curtis y en distancia euclidiana de datos transformados con Hellinger, complementadas con NMDS. Posteriormente se aplican pruebas PERMANOVA para contrastar diferencias composicionales entre grupos de exposición y PERMDISP para evaluar la homogeneidad de dispersión entre grupos. Esta última comprobación es esencial, ya que una PERMANOVA significativa solo puede interpretarse de forma más estricta como diferencia de localización cuando no está acompañada por heterogeneidad marcada en la dispersión interna.

⸻

Aportación esperada

El proyecto busca establecer si la huella espacial de la infraestructura petrolera offshore se asocia de manera detectable con la estructura ecológica y productiva de las pesquerías artesanales del litoral norte de Rio Grande do Norte. En particular, pretende determinar si el gradiente de exposición a plataformas deja una señal consistente en la magnitud de las capturas, en la diversidad de especies desembarcadas y en la composición de los ensamblajes, al tiempo que incorpora el contexto espacial de áreas protegidas como elemento complementario de interpretación.

La aportación esperada es doble. Desde una perspectiva científica, el proyecto proporciona evidencia espacialmente explícita sobre cómo una infraestructura industrial marina puede asociarse con cambios en sistemas pesqueros artesanales costeros. Desde una perspectiva aplicada, los resultados pueden informar decisiones sobre delimitación de zonas de exclusión, evaluación de impactos acumulativos, compensación a comunidades pesqueras afectadas y articulación entre infraestructura offshore y planificación espacial marina en contextos de alta dependencia social de la pesca artesanal.