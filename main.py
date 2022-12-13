import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
random.seed(1)

matplotlib.use("TkAgg")

def predict(A, B, Q, u_t, mu_t, P):
    # Оценка одношагового прогнозирования
    predicted_mu = A @ np.vstack(mu_t) + B * u_t
    # ковариационная матрица ошибки одношагового прогнозирования
    predicted_Sigma = A @ P @ A.T + Q
    return predicted_mu, predicted_Sigma


def update(H, R, z, predicted_mu, predicted_Sigma, sigm):
    # predicted_mu - это x k+1
    # Определить вектор обновления
    #eps
    residual_mean = np.vstack(z) - H @ predicted_mu
    L = np.exp(-((residual_mean).T @ np.linalg.inv(R) @ residual_mean)/(2*sigm**2))
    # Вычислить матричный коэффициент усиления Калмана
    residual_covariance = L * H @ predicted_Sigma @ H.T + R
    kalman_gain = L * predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)
    updated_mu = predicted_mu + kalman_gain @ residual_mean
    updated_Sigma = predicted_Sigma - kalman_gain @ H @ predicted_Sigma
    return updated_mu, updated_Sigma

def shot_noise():
    r = random.randint(0,22)
    if r/7 == 3 and r/3 == 7:
        k = random.randint(-70,70)
    else:
        k = 0

    return [k, k]
    #k = np.random.choice(a=[0, 25, 70, 100], p=[0.9725, 0.015, 0.01, 0.0025])
    #return [k, k]

#def rss(gtx,gty,mtx,mty): не работает, переделать исходя из выводов gtx,gtx,mtx,mty
    #return (gtx-mtx)**2 + (gty+mty)**2


# Истинное положение
dt = 0.1
sigm = 5
tetta = np.pi/3
num_steps = 150 # Кол-во измерений
# Считаем истинные значения
pos_x = []
pos_y = []
ground_truth_states = np.zeros((num_steps,4))
pos_x.append(1)
pos_y.append(1)
ground_truth_states[0][0] = position_x = 1.0
ground_truth_states[0][1] = position_y = 1.0
ground_truth_states[0][2] = velocity_x = 1.0
ground_truth_states[0][3] = velocity_y = 1.0
acceleration_x = np.sin(tetta)
acceleration_y = np.cos(tetta)
for i in range(1, num_steps):
    position_y = position_y + velocity_y * dt + (acceleration_y * dt ** 2) / 2.0
    position_x = position_x + velocity_x * dt + (acceleration_x * dt ** 2) / 2.0
    pos_x.append(position_x), pos_y.append(position_y)
    velocity_x = velocity_x + acceleration_x * dt
    velocity_y = velocity_y + acceleration_y * dt

ground_truth_states = np.stack((pos_x,pos_y,np.zeros(num_steps),np.zeros(num_steps)), axis=1)
# Начальное значение вектора состояний
mu_0 = np.array([1, 1, 0, 0])
# Начальное состояние вектора ковариации P
Sigma_0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
# Начальное значение вектора управляющих воздействий (у нас почему то это число)
u_t = 1

# Сделать рассчёт, а не константы?
A = np.array([[1, 0, dt, 0],       # Матрица процесса
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0],       # Матрица управления
              [0],
              [dt*np.sin(tetta)],
              [dt*np.sin(tetta)]])
Q = np.array([[0.5, 0, 0, 0],   # Матрица для ковариации (для вектора щума. ОЦЕНКА ПРЕДСКАЗАНИЙ ДЛЯ 1 ШАГА)
              [0, 0.5, 0, 0],
              [0, 0, 0.1, 0],
              [0, 0, 0, 0.1]])
H = np.array([[1, 0, 0, 0],       # Матрица наблюдений
              [0, 1, 0, 0]])
R = np.array([[5, 0],  # Матрица для ковариации (для вектора ошибки. ИСПОЛЬЗУЕТСЯ НА ПОДГОТОВИТЕЛЬНОМ ШАГЕ)
              [0, 5]])

# Для графиков
f_states = []
m_states = []

# Цикл фильтрации
mu_current = mu_0.copy()
Sigma_current = Sigma_0.copy()
for i in range(num_steps):
    # Одношаговое предсказание
    predicted_mu, predicted_Sigma = predict(A, B, Q, u_t, mu_current, Sigma_current)

    # Получение данных с датчиков (мы их зашумляем)
    measurement_noise = np.random.multivariate_normal(mean=np.array([0, 0]), cov=R)
    new_measurement = np.dot(H,ground_truth_states[i]) + measurement_noise + shot_noise()

        # Фильтрация
    mu_current, Sigma_current = update(H, R, new_measurement, predicted_mu, predicted_Sigma, sigm)

        # Для графиков
    m_states.append(new_measurement)
    f_states.append(mu_current)

m_states = np.array(m_states)  # Значения от датчиков
f_states = np.array(f_states)  # Отфильтрованные значения
# Графики
fig, axs = plt.subplots(2)
t = [2*i for i in range(num_steps)]
axs[0].set_title('Координаты по X')
axs[0].plot(t, ground_truth_states[:, 0])
axs[0].plot(t, m_states[:, 0])
axs[0].plot(t, f_states[:, 0])
axs[1].set_title('Координаты по Y')
axs[1].plot(t, ground_truth_states[:, 1])
axs[1].plot(t, m_states[:, 1])
axs[1].plot(t, f_states[:, 1])
plt.show()
