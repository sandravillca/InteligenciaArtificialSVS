import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

def run(episodes, learning_rate=0.9, discount_factor=0.9, c=1.0, is_training=True, render=False):
    
    # Inicializa el entorno MountainCar-v0 de Gymnasium
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Dividir la posición y velocidad en 20 segmentos utilizando np.linspace()
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Posición entre -1.2 y 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Velocidad entre -0.07 y 0.07

    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # Inicializar la tabla Q con: 20x20x3 array (3 acciones posibles)
        n = np.ones((len(pos_space), len(vel_space), env.action_space.n))  # INICIALIZAR EL CONTADOR DE ACCIONES CON 1 para evitar divisiones por cero
    else:
        with open('mountain_car.pkl', 'rb') as f:
            q = pickle.load(f)
    
    rng = np.random.default_rng()   # Generador de números aleatorios
    rewards_per_episode = np.zeros(episodes)    

    for i in range(episodes):
        
        state = env.reset()[0]      # posición inicial, velocidad inicial siempre 0
        state_p = np.digitize(state[0], pos_space)      # np.digitize() nos da a qué segmento pertenece el valor 
        state_v = np.digitize(state[1], vel_space)      

        terminated = False          # True cuando se alcanza el objetivo

        rewards = 0

        # Bucle para cada paso dentro de un episodio
        while not terminated and rewards > -1000:  # Penalización: se obtiene -1 por cada acción que toma el coche

            # Calcula A_t usando UCB
            ucb_values = q[state_p, state_v, :] + c * np.sqrt(np.log(i + 1) / n[state_p, state_v, :])
            action = np.argmax(ucb_values)

            new_state, reward, terminated, _, _ = env.step(action)  # Pasar la acción al entorno
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                # Actualizar la tabla Q y el contador de acciones
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )
                n[state_p, state_v, action] += 1 #CONTADOR DE ACCIONES

            state = new_state       # Actualiza el estado para el siguiente paso
            state_p = new_state_p   # Actualiza los segmentos del estado actual
            state_v = new_state_v

            rewards += reward       # Actualiza la recompensa

        rewards_per_episode[i] = rewards

    env.close()

    # Guardar la tabla Q al finalizar el entrenamiento
    if is_training:
        with open('mountain_car.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Calcular recompensas medias móviles
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(mean_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa media')
    plt.title('Recompensa media por episodio')
    plt.grid(True)
    plt.savefig('mountain_car_ucb.png')
    plt.show()

if __name__ == '__main__':
    run(5000, is_training=True, render=False)  # Entrenamiento
    #run(10, is_training=False, render=True)    # Evaluación
