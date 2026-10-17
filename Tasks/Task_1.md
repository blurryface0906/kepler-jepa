# TASK 1: Реализация физического генератора орбит, PyTorch-датасетов и модуля спектрального анализа

## Цель задачи
Создать базовую инфраструктуру проекта `kepler-jepa-verification`: математически точный генератор кеплеровских траекторий (аналитический решатель уравнения Кеплера и симплектический интегратор Верле), генератор выборок (`Dataset-Circ`, `Dataset-Ellip`, `Dataset-Counterfactual`), модуль спектрального оценивания частоты (`S1`, `S2`, `S3a`, `S3b`) и скрипт автоматической валидации `tests/verify_task1.py`.

---

## 1. Требования к структуре файлов
Создать следующую структуру директорий и файлов:

```text
kepler-jepa-verification/
├── src/
│   ├── __init__.py
│   ├── simulator.py
│   ├── dataset.py
│   └── spectral.py
├── tests/
│   └── verify_task1.py
└── requirements.txt
```

В `requirements.txt` зафиксировать: `numpy`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `tabulate`.

---

## 2. Спецификация модуля `src/simulator.py`
Все вычисления в симуляторе должны производиться строго в `np.float64` при гравитационном параметре $\mu = GM = 1.0$ и шаге дискретизации $\Delta t = 0.01$.

### 2.1. Функция `generate_kepler_analytical(a, e, t, mu=1.0, omega_arg=0.0, M0=0.0) -> np.ndarray`
- **Входы:**
  - `a`: `np.ndarray` формы `(N, 1)` — большие полуоси орбит.
  - `e`: `np.ndarray` формы `(N, 1)` — эксцентриситеты ($0 \le e < 1$).
  - `t`: `np.ndarray` формы `(1, T)` — временная сетка $t_k = k \Delta t$.
  - `mu`: `float = 1.0`.
  - `omega_arg`: `np.ndarray` формы `(N, 1)` или `float` — аргумент перицентра $\varpi \in [0, 2\pi)$.
  - `M0`: `np.ndarray` формы `(N, 1)` или `float` — начальная средняя аномалия $M_0$ ($M_0 = 0$ соответствует старту в перигелии, $M_0 = \pi$ — в афелии).
- **Алгоритм:**
  1. Истинная средняя частота: $n = \sqrt{\mu / a^3}$.
  2. Средняя аномалия: $M(t) = n \cdot t + M_0$.
  3. Решение уравнения Кеплера $M = E - e \sin E$ методом Ньютона (8 итераций, старт $E^{(0)} = M$):
     $$E^{(k+1)} = E^{(k)} - \frac{E^{(k)} - e \sin E^{(k)} - M}{1 - e \cos E^{(k)}}.$$
  4. Вычисление координат и скоростей в плоскости орбиты:
     $$r = a(1 - e \cos E), \quad x_{\text{orb}} = a(\cos E - e), \quad y_{\text{orb}} = a\sqrt{1 - e^2}\sin E,$$
     $$v_{x,\text{orb}} = -\frac{a^2 n}{r}\sin E, \quad v_{y,\text{orb}} = \frac{a^2 n}{r}\sqrt{1 - e^2}\cos E.$$
  5. Поворот векторов $(x_{\text{orb}}, y_{\text{orb}})$ и $(v_{x,\text{orb}}, v_{y,\text{orb}})$ на угол `omega_arg`.
- **Выход:** массив `trajectories` формы `(N, T, 4)` с каналами `[rx, ry, vx, vy]` в `np.float64`.

### 2.2. Функция `integrate_verlet(r0, v0, T_steps, dt=0.01, mu=1.0) -> np.ndarray`
- **Входы:** `r0` формы `(N, 2)`, `v0` формы `(N, 2)`, `T_steps: int`, `dt: float = 0.01`, `mu: float = 1.0`.
- **Алгоритм:** Симплектическая схема Velocity Verlet для ускорения $\mathbf{g}(\mathbf{r}) = -\frac{\mu}{\|\mathbf{r}\|^3}\mathbf{r}$:
  $$\mathbf{v}_{t+1/2} = \mathbf{v}_t + \frac{1}{2}\mathbf{g}(\mathbf{r}_t)\Delta t, \quad \mathbf{r}_{t+1} = \mathbf{r}_t + \mathbf{v}_{t+1/2}\Delta t, \quad \mathbf{v}_{t+1} = \mathbf{v}_{t+1/2} + \frac{1}{2}\mathbf{g}(\mathbf{r}_{t+1})\Delta t.$$
- **Выход:** массив `trajectories` формы `(N, T_steps, 4)` с каналами `[rx, ry, vx, vy]` в `np.float64`.

### 2.3. Функция `compute_orbital_invariants(trajectories, mu=1.0) -> dict`
- По траектории `(N, T, 4)` вычисляет в каждый момент времени:
  - Удельную механическую энергию: $E(t) = \frac{v_x^2 + v_y^2}{2} - \frac{\mu}{\sqrt{r_x^2 + r_y^2}}$.
  - Удельный момент импульса: $L_z(t) = r_x v_y - r_y v_x$.
  - Эффективную большую полуось: $a_{\text{eff}}(t) = -\frac{\mu}{2 E(t)}$.
- Возвращает словарь с максимальным относительным дрейфом энергии $\max_{t} \frac{|E(t) - E(0)|}{|E(0)|}$ и момента импульса по каждой орбите.

---

## 3. Спецификация модуля `src/dataset.py`
Реализовать генераторы трех конфигураций данных (с фиксируемым `seed`) и обертку `KeplerWindowDataset(torch.utils.data.Dataset)`:

1. **`build_dataset_circ(split, seed=42)`**:
   - $e = 0.0$, $\varpi = 0.0$, $M_0 \sim \mathcal{U}[0, 2\pi)$.
   - Сплиты:
     - `train`: $N=200$, логарифмически равномерная сетка (`np.geomspace`) $a \in [0.5, 5.0]$.
     - `test_id`: $N=50$, логарифмически равномерная сетка $a \in [0.5, 5.0]$ (сдвинутая относительно `train`).
     - `test_ood`: $N=50$ (25 орбит в $[0.25, 0.45]$ и 25 орбит в $[5.5, 8.0]$).
   - Длина траектории $T = 1024$ шага при $\Delta t = 0.01$.

2. **`build_dataset_ellip(split, seed=42)`**:
   - Те же сетки $a$, но $e \sim \mathcal{U}[0.2, 0.6]$, $\varpi \sim \mathcal{U}[0, 2\pi)$.
   - Для тестовых сплитов (`test_id`, `test_ood`) генерировать две отдельные версии траекторий:
     - `perihelion_init`: фаза центрирована так, чтобы середина стартового контекстного окна $x_{1:L}$ ($L=16$) попадала строго в перигелий ($M = 0$).
     - `aphelion_init`: середина стартового окна $x_{1:L}$ попадает строго в афелий ($M = \pi$).

3. **`build_dataset_counterfactual(seed=42)`**:
   - Сетка $a_i \in [0.5, 5.0]$ ($N=50$), старт в точке $\mathbf{r}_0 = (a_i, 0)$, начальная скорость $\mathbf{v}_0 = (0, \alpha_i \sqrt{\mu / a_i})$, где $\alpha_i$ равномерно распределены в $[0.75, 1.15]$ (исключая $\alpha=1.0$).
   - Траектория интегрируется через `integrate_verlet` на $T = 2048$ шагов.
   - Сохраняются метаданные: начальный радиус $a_i$, коэффициент $\alpha_i$, инерционная частота $\omega_{\text{init}} = \alpha_i \sqrt{\mu} a_i^{-1.5}$, истинная большая полуось $a_{\text{true}} = \frac{a_i}{2 - \alpha_i^2}$ и истинная гравитационная частота $\omega_{\text{true}} = \sqrt{\mu} a_{\text{true}}^{-1.5}$.

4. **Класс `KeplerWindowDataset`**:
   - Принимает массив траекторий `(N, T, C)` (с возможностью выбора `channels=2` для `[rx, ry]` или `channels=4` для `[rx, ry, vx, vy]`), длину контекста $L=16$ и максимальный горизонт $H=128$.
   - Возвращает кортеж `(context_window, target_windows, meta)` в `torch.float32`, где `context_window` имеет форму `(L, C)`, а `target_windows` содержит целевые окна длины $L$ для шагов $k = 1, \dots, H$ (форма `(H, L, C)`).

---

## 4. Спецификация модуля `src/spectral.py`
Все спектральные функции принимают многомерную траекторию `Z: np.ndarray` формы `(T_eval, d)` и шаг `dt: float = 0.01`.

1. **`project_to_2d_mode(Z: np.ndarray) -> np.ndarray`**:
   - Центрирует `Z` вдоль оси времени и через `sklearn.decomposition.PCA(n_components=2)` (или `np.linalg.svd`) возвращает ведущую 2D-моду `U` формы `(T_eval, 2)`.
   - Если вход уже двумерный (`d == 2`), возвращает центрированный массив без потери ранга.

2. **Четыре метода извлечения частоты $\hat{\omega}$ (возвращают `float` в рад/ед.вр.):**
   - `freq_s1_discrete_fft(U, dt)` — максимум суммарной периодограммы $|\text{rfft}(u_1)|^2 + |\text{rfft}(u_2)|^2$ (нулевая частота обнуляется).
   - `freq_s2_interpolated_fft(U, dt)` — умножение на окно Ханна `np.hanning(T)` и параболическая интерполяция логарифма спектральной мощности по трем соседним бинам $(m^*-1, m^*, m^*+1)$:
     $$\delta = \frac{1}{2}\frac{\ln S_{m^*-1} - \ln S_{m^*+1}}{\ln S_{m^*-1} - 2\ln S_{m^*} + \ln S_{m^*+1}}, \quad \hat{\omega} = \frac{2\pi(m^* + \delta)}{T \cdot dt}.$$
   - `freq_s3a_dmd_koopman(U, dt)` — метод Exact DMD (оператор Купмана) с задержкой по времени (Hankel / Time-Delay DMD или точный 2D линейный оператор перехода $A = U_{2:T}^T (U_{1:T-1}^T)^\dagger$): находит комплексные собственные значения матрицы перехода $\lambda$ и возвращает $\hat{\omega} = \frac{|\text{Im}(\ln \lambda_{\max})|}{dt}$.
   - `freq_s3b_phase_unwrap(U, dt)` — извлекает фазу $\phi(t) = \text{np.unwrap}(\text{np.arctan2}(u_2(t), u_1(t)))$ и находит $\hat{\omega} = |\text{slope}|$ линейной регрессией $\phi(t)$ по времени $t$ методом МНК (`np.polyfit(t, phi, 1)`).

3. **`evaluate_kepler_regression(semi_major_axes, omegas) -> dict`**:
   - Выполняет линейную регрессию $\log \hat{\omega}_i = \hat{\beta}_1 \log a_i + \hat{\beta}_0$.
   - Возвращает словарь: `{"slope": float, "slope_abs_error": float, "r2": float, "one_minus_r2": float, "log10_one_minus_r2": float}`.

---

## 5. Валидационный скрипт `tests/verify_task1.py` и Критерии приемки (Acceptance Criteria)
Написать и запустить скрипт `tests/verify_task1.py`, который проверяет 4 обязательных теста и выводит сводную таблицу в консоль:

1. **Тест 1 (Точность решателя Кеплера vs Интегратора Верле):**
   - На `Dataset-Ellip (test_id)` ($e \in [0.2, 0.6]$, $T=1024$) максимальный относительный дрейф энергии в аналитическом решателе должен быть $< 10^{-13}$, а в симплектическом интеграторе Верле при $\Delta t = 0.01$ — $< 10^{-4}$ (без монотонного векового тренда).
2. **Тест 2 (Проверка Гипотезы H2 на истинных круговых траекториях `Dataset-Circ`, $T_{\text{eval}} = 256$):**
   - Прогнать координаты `[rx, ry]` длины $T_{\text{eval}} = 256$ через методы `S1`, `S2`, `S3a`, `S3b` для орбит $a \in [0.5, 2.0]$.
   - **Критерий:** Для `S1 (Discrete FFT)` показатель $R^2$ ограничен ошибкой квантования сетки ($1 - R^2 > 10^{-5}$), тогда как для `S3a (DMD/Koopman)` и `S3b (Phase Unwrap)` должно выполняться $|\hat{\beta}_1 - (-1.5)| < 10^{-4}$ и $R^2 > 0.9999999$ ($1 - R^2 < 10^{-7}$).
3. **Тест 3 (Ловушка короткого окна на эллиптических орбитах `Dataset-Ellip`):**
   - Измерить локальную угловую скорость методом `S3b` на коротком окне $L = 64$ вблизи перигелия (`perihelion_init`) и афелия (`aphelion_init`) против полного окна. Показать численное расхождение мгновенных частот $\hat{\omega}_{\text{peri}} > \omega_0 > \hat{\omega}_{\text{ap}}$ при сохранении наклона по большой полуоси.
4. **Тест 4 (Валидация `Dataset-Counterfactual`):**
   - Проверить на длинной траектории Верле ($T = 2048$), что спектральный пик `S2` / `S3b` сходится к гравитационной частоте $\omega_{\text{true}} = \sqrt{\mu}\left(\frac{2-\alpha^2}{a}\right)^{1.5}$ ($R^2 > 0.999$), а на первых $L=16$ шагах мгновенная частота `S3b` совпадает с инерционной $\omega_{\text{init}} = \alpha \sqrt{\mu} a^{-1.5}$ ($R^2 > 0.9999$).

---

## Формат отчета о выполнении
Предоставить:
1. Полный исходный код файлов `src/simulator.py`, `src/dataset.py`, `src/spectral.py` и `tests/verify_task1.py`.
2. Полный текстовый вывод (stdout) запуска `python -m tests.verify_task1` с таблицами результатов всех 4 тестов.
task_1_simulator_spectral.md
Показан объект "task_1_simulator_spectral.md".
