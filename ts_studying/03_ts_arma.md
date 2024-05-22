# ARMA(p,q)

Модель ARMA(p, q) представляет собой соединение двух моделей:
- AR(P) - авторегрессии на знанениях временного ряда
- MA(q) - авторегрессии на ошибках первой

AR(p) пытается предсказать "значение" временного ряда, а MA(q) пытается поймать шоковые явления, наблюдаемые в оставшемся случайном шуме.

$y_t = \sum_{i=1}^P a_i x_{t-i} + \sum_{i=1}^Q b_i \epsilon_{t-i} + \epsilon_t + c$

## Сделаем симуляцию процесса ARMA:

[statsmodels.tsa.arima_process.arma_generate_sample](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.arma_generate_sample.html#statsmodels.tsa.arima_process.arma_generate_sample)

[Autoregressive Moving Average (ARMA): Artificial data](https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_1.html)

[statsmodels.tsa.arima.model.ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)



```python

statsmodels.tsa.arima_process.arma_generate_sample(
    ar, # Коэффициент для полинома авторегрессионного запаздывания, включая нулевую задержку.
    ma, # Коэффициент для полинома задержки скользящего среднего, включая нулевую задержку.
    nsample,
    scale=1,
    distrvs=None,
    axis=0,
    burnin=0
)

```
- *nsample*: (*int* or *tuple*) - Если nsample является целым числом, то создается временная серия длиной 1d. Если nsample является кортежем, создает временной ряд размерности len(nsample), в котором время индексируется по входной переменной axis. Все серии есть, если только не distrvs генерирует зависимые данные.
- *scale*: (*float*) - Стандартное отклонение шума.
- *distrvs*: (*function*) - Функция, которая генерирует случайные числа и принимает их size в качестве аргумента. По умолчанию используется np.random.standard_normal.
- *axis*: (*int*) - см. *nsample*.
- *burnin*: (*int*) - Количество наблюдений в начале выборки уменьшается. Используется для уменьшения зависимости от начальных значений.

Возвращает *ndarray*: случайные образцы и процесса ARMA.

Example:

```python

# импорты

from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

# генерируем массив ARMA

n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.5, -0.25])
betas = np.array([0.5, -0.3])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = sm.tsa.arima_process.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)

# обучаем модиль ARMA

arma22 = pd.Series(arma22)
arma_model = ARIMA(arma22, order=(2, 0, 2), trend='n').fit()
print(arma_model.summary())

# Визуализация результатов + 100 предсказанных значений:

with plt.style.context(style='bmh'):
    plt.figure(figsize=(14,8))
    ax = plt.axes()
    plot_predict(arma_model, start=4900, end=5100, ax=ax)



```