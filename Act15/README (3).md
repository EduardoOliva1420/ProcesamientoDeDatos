# Reporte: Cómputo Evolutivo — Algoritmos Genéticos

> **Referencia:** Kuri, Á. (2002). *Algoritmos genéticos*. México D.F.: Instituto Politécnico Nacional.  
> Recuperado de [elibro.net](https://elibro.net/es/ereader/mayab/71925?page=14)

---

## 1. La naturaleza como optimizadora

La naturaleza lleva millones de años resolviendo problemas de optimización de una forma sorprendente: a través de la evolución. Los seres vivos deben adaptarse continuamente a entornos cambiantes y competitivos, y solo sobreviven y se reproducen aquellos que lo hacen mejor. Este proceso, descrito por Charles Darwin, es en esencia un mecanismo de búsqueda y optimización.

Kuri (2002) señala que la selección natural actúa como una función objetivo implícita: los individuos con rasgos más favorables tienen mayor probabilidad de dejar descendencia. Con el paso de las generaciones, la población en su conjunto mejora su adaptación al entorno. Esto es precisamente lo que los **Algoritmos Genéticos (AG)** intentan imitar computacionalmente: usar la presión selectiva del ambiente para guiar la búsqueda de soluciones óptimas en espacios de búsqueda complejos.

Lo poderoso de este enfoque es que **no requiere conocer la forma del espacio de búsqueda**. El algoritmo explora múltiples regiones en paralelo (a través de la población) y converge naturalmente hacia las mejores soluciones, igual que una especie converge hacia rasgos adaptativos.

---

## 2. Un poco de biología

Para entender los Algoritmos Genéticos es necesario conocer los conceptos biológicos que los inspiran:

| Concepto biológico | Concepto en AG |
|---|---|
| Individuo / organismo | Solución candidata |
| Cromosoma | Cadena de genes (cadena de bits, por ejemplo) |
| Gen | Bit o valor en una posición del cromosoma |
| Aptitud (fitness) | Valor de la función objetivo evaluada en esa solución |
| Selección natural | Proceso de elegir los individuos más aptos para reproducirse |
| Cruzamiento (crossover) | Combinación de dos cromosomas para producir un hijo |
| Mutación | Cambio aleatorio en uno o más genes |
| Generación | Una iteración del ciclo evolutivo |

### Estructura del cromosoma

En la representación más clásica (Kuri, 2002), el cromosoma es una **cadena binaria** donde cada bit es un gen. Una posible solución al problema se codifica en este string. Por ejemplo, para un problema de 8 variables binarias:

```
Cromosoma: 1 0 1 1 0 0 1 1
            ↑           ↑
           gen 0       gen 7
```

### El ciclo evolutivo

El proceso de un Algoritmo Genético sigue este ciclo:

```
Inicializar población aleatoria
         ↓
   Evaluar aptitud
         ↓
¿Se cumple criterio de paro? → Sí → Retornar mejor individuo
         ↓ No
    Selección
         ↓
   Cruzamiento
         ↓
    Mutación
         ↓
Nueva generación → volver a evaluar aptitud
```

---

## 3. Implementación en Python

### Clase `Individuo`

Representa una solución candidata. Su cromosoma es una lista de bits generada aleatoriamente.

```python
class Individuo:
    def __init__(self, longitud_cromosoma: int, generacion: int = 0):
        self.longitud_cromosoma = longitud_cromosoma
        self.generacion = generacion
        self.cromosoma = []
        self.aptitud = 0.0
        self.inicializar()

    def inicializar(self):
        self.cromosoma = [random.randint(0, 1) for _ in range(self.longitud_cromosoma)]

    def calcular_aptitud(self):
        self.aptitud = sum(self.cromosoma)  # Problema OneMax
        return self.aptitud

    def mutar(self, prob_mutacion: float = 0.01):
        for i in range(self.longitud_cromosoma):
            if random.random() < prob_mutacion:
                self.cromosoma[i] = 1 - self.cromosoma[i]

    def obtener_gen(self, indice: int) -> int:
        return self.cromosoma[indice]

    def __repr__(self) -> str:
        genes = ''.join(map(str, self.cromosoma))
        return f"Individuo(gen={self.generacion}, cromosoma={genes}, aptitud={self.aptitud})"
```

### Clase `Población`

Contiene y gestiona un conjunto de individuos. Aplica selección por torneo, cruzamiento de un punto y mutación.

```python
class Poblacion:
    def __init__(self, tamanio: int, longitud_cromosoma: int, tasa_mutacion: float = 0.01):
        self.tamanio = tamanio
        self.tasa_mutacion = tasa_mutacion
        self.generacion_actual = 0
        self.individuos = []
        self._longitud_cromosoma = longitud_cromosoma
        self.inicializar_poblacion()

    def inicializar_poblacion(self):
        self.individuos = [
            Individuo(self._longitud_cromosoma, generacion=0)
            for _ in range(self.tamanio)
        ]
        for ind in self.individuos:
            ind.calcular_aptitud()

    def seleccion(self) -> 'Individuo':
        competidores = random.sample(self.individuos, min(3, len(self.individuos)))
        return max(competidores, key=lambda ind: ind.aptitud)

    def cruzar(self, padre1: 'Individuo', padre2: 'Individuo') -> 'Individuo':
        punto = random.randint(1, self._longitud_cromosoma - 1)
        hijo = Individuo(self._longitud_cromosoma, generacion=self.generacion_actual + 1)
        hijo.cromosoma = padre1.cromosoma[:punto] + padre2.cromosoma[punto:]
        return hijo

    def evolucionar(self):
        nueva_generacion = []
        for _ in range(self.tamanio):
            padre1 = self.seleccion()
            padre2 = self.seleccion()
            hijo = self.cruzar(padre1, padre2)
            hijo.mutar(self.tasa_mutacion)
            hijo.calcular_aptitud()
            nueva_generacion.append(hijo)
        self.individuos = nueva_generacion
        self.generacion_actual += 1

    def mejor_individuo(self) -> 'Individuo':
        return max(self.individuos, key=lambda ind: ind.aptitud)

    def __repr__(self) -> str:
        mejor = self.mejor_individuo()
        return (
            f"Población(generación={self.generacion_actual}, "
            f"tamaño={self.tamanio}, mejor_aptitud={mejor.aptitud})"
        )
```

---

## 4. Diagrama UML de clases

```
┌─────────────────────────────────┐          ┌──────────────────────────────────────┐
│           Individuo             │          │              Población               │
├─────────────────────────────────┤  1    1..* ├──────────────────────────────────────┤
│ - cromosoma: list               │◆─────────►│ - individuos: list[Individuo]        │
│ - aptitud: float                │          │ - tamanio: int                       │
│ - longitud_cromosoma: int       │          │ - generacion_actual: int             │
│ - generacion: int               │          │ - tasa_mutacion: float               │
├─────────────────────────────────┤          ├──────────────────────────────────────┤
│ + inicializar()                 │          │ + inicializar_poblacion()            │
│ + calcular_aptitud()            │          │ + seleccion()                        │
│ + mutar(prob_mutacion)          │          │ + cruzar(padre1, padre2)             │
│ + obtener_gen(indice)           │          │ + evolucionar()                      │
│ + __repr__()                    │          │ + mejor_individuo()                  │
└─────────────────────────────────┘          │ + __repr__()                         │
                                             └──────────────────────────────────────┘
```

La relación es de **composición**: una `Población` está formada por uno o más `Individuo`s. Si la población deja de existir, los individuos también.

---

## 5. Reflexión personal

La lectura de Kuri (2002) permite entender que los Algoritmos Genéticos no son simplemente heurísticas aleatorias, sino que replican de forma elegante el mecanismo más poderoso de optimización que ha existido: la evolución biológica. La clave está en que la búsqueda no es ciega: la aptitud guía hacia mejores soluciones, el cruzamiento explota combinaciones prometedoras y la mutación evita el estancamiento en óptimos locales.

Las clases `Individuo` y `Población` encapsulan de forma natural estos conceptos: cada individuo "sabe" qué es (su cromosoma) y qué tan bueno es (su aptitud), mientras que la población "sabe" cómo evolucionar colectivamente.

---

*Elaborado con base en: Kuri, Á. (2002). Algoritmos genéticos. IPN.*
