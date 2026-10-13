# Guía de entrevista: post-training con GRPO y recompensas verificables

**Resultados:** pendientes de entrenamiento y evaluación final. Completa las métricas con los artefactos del experimento; no presentes los objetivos como resultados obtenidos.

## Presentación de tres minutos

> Quería estudiar una técnica de post-training con RL que pudiera evaluar de forma objetiva en mi propio hardware. Construí un experimento de Countdown con Qwen2.5-1.5B-Instruct: el modelo recibe tres o cuatro números y un objetivo, y devuelve una expresión que debe usar cada número exactamente una vez y alcanzar el objetivo.
>
> La recompensa es verificable: un parser restringido valida la expresión y la evalúa con fracciones exactas. El GRPO principal recibe 1,0 si alcanza el objetivo, 0,2 si devuelve una expresión válida pero incorrecta y 0 si incumple el formato. Entreno adaptadores LoRA con Dr. GRPO desde el checkpoint Instruct, sin añadir mi propia fase SFT antes del RL. Comparo contra el modelo original, una ablación que premia solo expresiones válidas y un control SFT que sí ve soluciones de entrenamiento.
>
> Genero problemas con un solucionador exacto y separo las particiones por conjunto de números para evitar que el mismo problema aparezca en train y test con otro orden. Evalúo una respuesta greedy en un test ID y separo los resultados por aridad. También tengo un test OOD con números mayores. Guardo las respuestas, recompensas, hashes, configuraciones y costes para poder auditar el resultado.
>
> La conclusión se limita a esta tarea sintética: no afirmo que el modelo haya adquirido razonamiento general ni que sea mejor que un algoritmo exacto. Las cifras finales son [resultado medido], con [número de semillas] semillas y [horas] GPU-horas. El hallazgo principal fue [hallazgo basado en los resultados].

Completa los campos entre corchetes cuando los datos reales estén disponibles.

## Explicación del ciclo de RL

1. El prompt presenta los números y el objetivo.
2. La política genera varias expresiones para el mismo problema.
3. Un verificador determinista asigna una recompensa, sin recurrir a un modelo juez.
4. GRPO compara las recompensas dentro del grupo y calcula ventajas relativas para actualizar la política LoRA.
5. La evaluación greedy mide si el cambio mejora la probabilidad de resolver el problema con una respuesta.

Dr. GRPO es la variante fijada en este proyecto: centra las recompensas dentro del grupo y usa un denominador fijo para la pérdida. Al explicar el resultado, menciona la versión de TRL y el valor real de `loss_type` registrado en la ejecución.

### Preguntas probables

**¿Qué representan la política, la acción y la recompensa?**

La política es el modelo condicionado por el problema. La acción es la respuesta generada token a token. La recompensa vale 1,0 si la expresión válida alcanza el objetivo, 0,2 si usa correctamente los números pero no llega al objetivo, y cero si la expresión no es válida. Un parser determinista con aritmética racional exacta calcula la recompensa.

**¿Qué aporta GRPO frente a un crítico separado?**

Un grupo de respuestas para el mismo prompt permite estimar ventajas relativas a partir de sus recompensas. Este diseño evita entrenar un modelo crítico por separado. Si todas las respuestas del grupo reciben la misma recompensa, hay poca señal comparativa; por eso se registra la proporción de grupos sin variación.

**¿Por qué Dr. GRPO?**

Es la variante fijada para normalizar la pérdida con un denominador constante, evitando que la longitud altere esa normalización como ocurre con otras formulaciones. Hay que poder señalar la implementación de TRL y su configuración concreta.

**¿Por qué usar este shaping?**

La corrección se puede decidir exactamente sin etiquetas subjetivas ni un evaluador aprendido. En el smoke inicial con recompensa binaria, el primer grupo de actualización recibió recompensa cero y tuvo gradiente cero; en una fase posterior aparecieron aciertos exactos escasos. El componente de validez añade una señal verificable cuando el modelo ya construye una expresión admisible, mientras que la recompensa completa sigue exigiendo alcanzar el objetivo.

**¿Por qué no premiar una explicación paso a paso?**

La explicación no define el éxito y sería más difícil de verificar. Añadir esa recompensa podría optimizar longitud o estilo sin mejorar la solución. El modelo debe devolver una expresión y el verificador juzga esa expresión exacta.

**¿Por qué no usar directamente el solucionador?**

Para resolver Countdown, el algoritmo exacto es una opción más apropiada. El objetivo del proyecto es estudiar post-training con una recompensa verificable, no sustituir al solucionador.

**¿LoRA es RL?**

No. LoRA es una forma eficiente de parametrizar y guardar los cambios. GRPO define el objetivo de optimización y la señal de recompensa.

## Datos, controles y evaluación

**¿Cómo evitaste fuga entre particiones?**

La partición se hace por multiconjunto de números, no por el orden en que se muestran. El mismo multiconjunto no se comparte entre train, validation y test ID, aunque se permuten los números o cambie el target. Los datos se congelan con hashes antes de entrenar, y test no se usa para elegir checkpoints ni hiperparámetros.

**¿Qué mide el test OOD?**

Solo cambia el rango de números: train e ID usan 1–30 y OOD usa 31–60. Es una prueba acotada de cambio de rango, no evidencia de generalización amplia ni un requisito de éxito.

**¿Para qué sirve la ablación de validez?**

Premia una expresión admisible que use los números correctamente, aunque no alcance el objetivo. Su comparación con la recompensa principal ayuda a distinguir si el modelo aprende el formato o la resolución del problema.

**¿Por qué incluir SFT en un proyecto de RL?**

Es una referencia supervisada sencilla. SFT usa las expresiones testigo de train y recibe más información que el GRPO, así que no es una comparación de idéntica información ni cómputo. Esa diferencia y los costes deben declararse.

**¿Qué significa “direct GRPO” si el modelo ya es Instruct?**

El checkpoint público ya fue afinado con instrucciones. “Directo” significa que el proyecto no añade una etapa SFT propia antes de GRPO; no significa que el modelo nunca recibiera post-training.

La métrica principal es exactitud greedy en test con una sola respuesta por problema. Validez, exactitud entre válidas, resultados por aridad y tipos de error explican cómo falla el sistema. Los resultados de cuatro respuestas muestreadas se presentan separados de greedy. El intervalo bootstrap pareado resume variación entre problemas; las diferencias entre semillas se presentan aparte.

## Preguntas difíciles y límites

**¿Qué pasa si GRPO no mejora?**

El resultado puede seguir siendo útil si la evaluación es reproducible: mostraría que esta recompensa, escala, prompt y presupuesto no bastaron con el protocolo definido. Revisaría variación de recompensas, errores y resultados por aridad, y reportaría la conclusión negativa o inconclusa. No cambiaría el protocolo después de mirar test.

**¿Y si SFT supera a GRPO?**

Es plausible porque SFT tiene las expresiones correctas y una señal de entrenamiento por token. Reportaría el resultado con su ventaja informativa y su coste. El experimento mide qué aporta esta configuración de RL; no presupone que tenga que vencer a SFT.

**¿Qué demuestra un resultado OOD más bajo?**

Una caída al cambiar el rango de entrada bajo esta partición. No demostraría una limitación general de razonamiento. Hay que analizarla junto con aridad, validez, errores y distribución sintética.

**¿Qué no demuestra el proyecto?**

No demuestra razonamiento general, transferencia a dominios reales, superioridad ante algoritmos exactos ni que RL sea siempre mejor que supervisión. Tampoco permite ocultar una semilla regresiva tras la media.

## Ficha de métricas para la entrevista

- Exactitud ID greedy del modelo original: **pendiente**.
- Exactitud ID greedy de GRPO por semilla: **pendiente**.
- Diferencia media y bootstrap pareado: **pendiente**.
- Exactitud OOD y resultados separados por aridad: **pendiente**.
- Validez de salida y grupos sin variación de recompensa: **pendiente**.
- Resultados de SFT y de la ablación: **pendiente**.
- Tiempo, memoria máxima, GPU y número de actualizaciones: **pendiente**.
- Un ejemplo de mejora y otro de regresión con respuestas crudas: **pendiente**.

No uses una cifra pendiente en el CV. Cuando haya resultados, una frase posible será: “Ajusté [modelo] con [método] y cambié la exactitud greedy de [A] a [B] en [N] problemas de prueba, con [K] semillas y [H] horas de GPU”. Si no mejora, describe el diseño, la evaluación y el aprendizaje medido sin afirmar una mejora.
