# Guía de entrevista: post-training con GRPO y recompensas verificables

Esta guía usa los resultados medidos del 14 de octubre de 2026. Las cifras del test son exactitud greedy sobre problemas congelados; los intervalos comparan problemas pareados y no miden variación entre semillas.

## Presentación de un minuto

> Quería probar post-training con RL en una tarea pequeña que pudiera verificar sin un modelo juez. Partí de Qwen2.5-1.5B-Instruct y entrené adaptadores LoRA con GRPO para resolver problemas Countdown. El modelo recibe tres o cuatro enteros y un objetivo, y debe devolver una sola expresión que use cada número una vez. Un parser seguro la evalúa con fracciones exactas.
>
> La recompensa principal da 1 si la expresión alcanza exactamente el objetivo, 0,2 si usa bien los números pero falla el objetivo y 0 si no cumple las reglas. En test ID, el modelo base resolvió 10 de 1.024 problemas y el checkpoint seleccionado de GRPO resolvió 16. La diferencia fue +0,59 puntos porcentuales, con intervalo bootstrap pareado del 95% de −0,39 a +1,56, así que no puedo afirmar una mejora estadísticamente clara. La validez sí subió de 18,4% a 65,7%.
>
> La validación a step 200 quedó 0,78 puntos por debajo de la base y no alcanzó el umbral de continuación de +3 puntos. Apliqué la regla predefinida, detuve las semillas 43 y 44 y reporto el resultado como un experimento primario de una sola semilla. Una ablación que premia solo la validez llegó a 100% de respuestas válidas y 11,4% de exactitud ID, pero resolvió 0 de 512 problemas OOD. El control SFT con expresiones testigo alcanzó 21,7% ID y 22,1% OOD; tiene acceso a etiquetas y no es una comparación de información equivalente.
>
> La conclusión es acotada: la recompensa verificable ayudó sobre todo a producir expresiones admisibles, pero esta configuración de GRPO no demostró una mejora fiable de exactitud. No afirmo razonamiento general ni que el modelo sea mejor que un solucionador exacto.

## Resultados para recordar

| Ejecución | ID greedy | ID válidas | OOD greedy | ID pass@4 |
|---|---:|---:|---:|---:|
| Qwen base | 10/1.024 (0,98%) | 18,4% | 2/512 (0,39%) | 2,7% |
| GRPO recompensa exacta, semilla 42 | 16/1.024 (1,56%) | 65,7% | 0/512 (0%) | 1,6% |
| GRPO solo validez, semilla 42 | 117/1.024 (11,43%) | 100% | 0/512 (0%) | 12,1% |
| SFT con testigos, semilla 42 | 222/1.024 (21,68%) | 97,6% | 113/512 (22,07%) | 32,0% |

Las diferencias ID pareadas frente a la base, con 10.000 remuestreos por problema, fueron:

- GRPO principal: +0,59 pp; IC bootstrap del 95% [−0,39, +1,56], n=1.024.
- Ablación de validez: +10,45 pp; IC bootstrap del 95% [+8,40, +12,50], n=1.024.
- Control SFT: +20,70 pp; IC bootstrap del 95% [+18,07, +23,34], n=1.024.

Los dos últimos intervalos son comparaciones secundarias con una sola semilla cada una. El bootstrap cuantifica variación entre problemas del test; no estima la variabilidad de repetir el entrenamiento con otra semilla. No se obtuvo una estimación primaria de tres semillas.

## Cómo funciona el entrenamiento

1. El prompt fija los números y el objetivo. El modelo genera cuatro respuestas para cada problema de entrenamiento.
2. El verificador restringe la sintaxis, comprueba la multiplicidad de los números y calcula el resultado con `Fraction`; nunca ejecuta código generado.
3. GRPO compara las recompensas dentro de cada grupo y usa esas diferencias para actualizar LoRA. No se entrena un crítico separado.
4. Se registra cuántos grupos tienen recompensas distintas: si todo el grupo recibe el mismo valor, hay poca señal relativa para la actualización.
5. Se elige el checkpoint con validación; el test congelado se consulta después. La métrica principal es una respuesta greedy por problema.

La configuración fija TRL 0.23.1, `loss_type=dr_grpo`, cuatro generaciones, `beta=0.01`, `epsilon=0.2`, LoRA r=16 y BF16. La GPU fue una NVIDIA GeForce RTX 4070 Laptop GPU con 8 GiB de VRAM. El ledger registró 15,21 GPU-horas acumuladas de un máximo de 72, incluyendo preparación, entrenamiento y evaluación.

## Preguntas probables

**¿Qué son la política, la acción y la recompensa?**

La política es el modelo condicionado por el problema; la acción es la expresión generada token a token. La recompensa principal es 1,0 para una expresión exacta, 0,2 para una expresión válida que no alcanza el objetivo y 0 para una respuesta inválida. El cálculo es determinista y usa aritmética racional exacta.

**¿Qué aporta GRPO frente a un crítico separado?**

Compara varias respuestas al mismo prompt para obtener ventajas relativas sin entrenar un modelo crítico. En el probe del checkpoint principal, 95,35% de los grupos tuvo alguna recompensa positiva, pero solo 27,13% tuvo recompensas no uniformes. Tener señal positiva no garantizó una mejora en exactitud de validación.

**¿Por qué Dr. GRPO?**

Es la variante fijada en TRL: normaliza la pérdida con un denominador fijo. El método de optimización se debe distinguir de LoRA, que solo parametriza y guarda la adaptación.

**¿Por qué añadir 0,2 a expresiones válidas incorrectas?**

Es una recompensa intermedia exacta para separar “escribe una respuesta admisible” de “resuelve el problema”. La ablación que premia solo validez sirve para observar esa diferencia. Medida en test, la validez llegó a 100%, pero la exactitud OOD siguió en cero: aprender el formato no bastó para resolver en el rango nuevo.

**¿Por qué no pedir una explicación paso a paso?**

La explicación no define el éxito y sería más difícil de verificar. Podría optimizar longitud o estilo. El contrato del proyecto pide una sola expresión y mide esa expresión.

**¿Por qué no usar directamente el solucionador exacto?**

Para resolver Countdown, el solucionador dinámico determinista es la herramienta adecuada. El objetivo es estudiar post-training con una recompensa verificable, no reemplazar al algoritmo exacto.

**¿Qué significa “direct GRPO” si el modelo ya es Instruct?**

El checkpoint público ya fue afinado con instrucciones. “Directo” significa que el proyecto no añadió su propia etapa SFT antes del RL.

**¿Cómo se evitó la fuga entre particiones?**

La partición ID se asigna por multiconjunto de números. No se comparte el mismo multiconjunto entre train, validación y test, aunque cambien el orden o el objetivo. Los test ID usan números 1–30 y OOD usa 31–60. El conjunto OOD cambia el rango, así que mide una transferencia estrecha.

**¿Por qué incluir un control SFT?**

Sirve de referencia supervisada y hace visible el valor de las etiquetas testigo. SFT ve una expresión correcta por problema de train; GRPO solo ve el problema y la recompensa. No tienen la misma información ni el mismo coste por token. SFT obtuvo la mejor exactitud y transfirió al rango OOD en este protocolo, pero eso no constituye una comparación justa de cantidad de información.

**¿Por qué no se entrenaron las tres semillas principales?**

La regla registrada antes de consultar test requería que el probe del primer seed superara a la base en al menos 3 pp de exactitud en la validación fija de 128 filas y que hubiera suficiente variación positiva de recompensas. A step 200, la exactitud quedó −0,78 pp frente a la base. Se conservaron el checkpoint y sus datos como resultado medido y no se gastó presupuesto en los seeds 43 y 44. Por eso la incertidumbre entre semillas permanece sin medir.

**¿El GRPO principal mejoró?**

La exactitud ID observada subió de 0,98% a 1,56%, pero son 6 aciertos adicionales de 1.024 y el intervalo pareado incluye cero. La validez subió 47,36 pp. La afirmación defendible es que el modelo produjo más respuestas admisibles; el experimento no demuestra una mejora fiable de resolución exacta.

**¿Puedes mostrar una respuesta cruda?**

En una fila ID con `[9, 2, 16]` y objetivo `41`, el modelo base generó `(9 * (16 - 2)) + 1`: el verificador la rechaza porque introduce un número adicional. GRPO generó `(9 * 2) + 16`, que usa los tres números una vez y vale `34`; por eso recibe la recompensa intermedia de 0,2, pero no resuelve el objetivo. Una solución exacta es `9 + (16 * 2)`. El ejemplo está guardado en los JSONL de predicciones y muestra por qué la validez y la exactitud se reportan por separado.

**¿Qué demuestra el resultado OOD?**

El GRPO principal y la ablación de validez no resolvieron casos OOD con una respuesta greedy. El SFT resolvió 113/512. OOD cambia solo el rango de números; no demuestra transferencia amplia.

## Frase para el CV

> Implementé y evalué post-training con recompensas verificables en Qwen2.5-1.5B-Instruct usando GRPO/Dr. GRPO y LoRA, un verificador aritmético exacto y controles SFT y de validez; en 1.024 problemas ID, el GRPO principal elevó las salidas válidas de 18,4% a 65,7%, mientras la exactitud fue 1,56% frente a 0,98% en base (diferencia +0,59 pp; IC pareado 95% −0,39 a +1,56).

Si prefieres una frase corta, elimina las métricas y di “medí mejoras de formato, pero no obtuve evidencia concluyente de mejora de exactitud”. No atribuyas al GRPO principal la exactitud de la ablación o del control SFT.

## Artefactos para enseñar

- [Informe completo y figuras](../results/report.md).
- [Protocolo congelado](protocol.md), con reglas de datos, selección, corte y evaluación.
- `results/summary.json`, métricas detalladas y `results/evaluations/`, predicciones crudas y parseadas.
- Manifiestos versionados en `results/run_evidence/`; incluyen revisión del modelo, configuración, equipo, métricas y checkpoint seleccionado. Los checkpoints y pesos locales se guardan fuera de Git.

Para cargar el adaptador principal, instala el entorno descrito en el README y usa `uv run rl-countdown predict --config configs/default.yaml --nums 2 3 7 --target 20 --adapter <directorio-del-adaptador>`. El paquete y su checksum están descritos en [el manifiesto de release](release-v0.1.0.md). Para regenerar las métricas necesitas ejecutar la evaluación sobre la misma revisión de modelo y los mismos datos congelados.
