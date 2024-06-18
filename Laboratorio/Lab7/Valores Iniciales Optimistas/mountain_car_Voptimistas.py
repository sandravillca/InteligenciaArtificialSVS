import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
Inicializamos los valores de q, en lugar de utilizar 0's para inicializar la tabla,
utilizamos 1 multiplicado por un determinado número que instará a que se busque una 
recompensa más alta, haciendo que explore todas las acciones posibles antes de encontrar
la acción más óptima 
"""

def run(episodes, is_training=True, render=False):
    
    #Inicializa el entorno MountainCar-v0 de Gymnasium
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    # Dividir la posición y velocidad en 20 segmentos utilizando np.linspace()
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Posición entre -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Velocidad entre -0.07 and 0.07
    
    

    if(is_training):
        initial_q_value = 10 # Inicializmaos Q con un valor optimista
        q = np.ones((len(pos_space), len(vel_space), env.action_space.n)) * initial_q_value# Inicializar la tabla q
    else:
        f = open('mountain_car_opt.pkl', 'rb')
        q = pickle.load(f)
        f.close()



    #Parámetros del algoritmo Q learning
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor. FACTOR DE DESCUENTO A LA RECOMPENSA
    epsilon = 1         # 1 = 100% random actions. PROBABILIDAD INICIAL DE EXPLORACIÓN
    epsilon_decay_rate = 2/episodes # epsilon decay rate
    rng = np.random.default_rng()   # random number generator
    rewards_per_episode = np.zeros(episodes)    





    for i in range(episodes):        
        state = env.reset()[0]      # posición inicial, velocidad inicial siempre 0
        state_p = np.digitize(state[0], pos_space)      #np.digitize() nos da a qué segmento pertenece el valor 
        state_v = np.digitize(state[1], vel_space)      

        terminated = False          # True when reached goal
        rewards=0


        #Bucle para cada paso dentro de un episodio.
        #Dos condiciones que terminan el juego: Alcanzar el objetivo ó Llamada a más de 1000 acciones
        while(not terminated and rewards>-1000):  #penalty structure: we get -1 for every action that the car takes


            if is_training and rng.random() < epsilon:
                # Elegir una acción random (0=conducir a la izquierda, 1=quedarse parado, 2=conducir a la derecha)
                action = env.action_space.sample()  #Exploración: selecciona una acción aleatoria
            else:
                action = np.argmax(q[state_p, state_v, :])  #Explotación: selecciona la mejor acción basada en Q-table


            new_state,reward,terminated,_,_ = env.step(action)  #Le pasamos la acción al entorno. El nuevo estado consiste de una posición y una velocidad
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                ##Actualiza la tabla Q con la nueva información obtenida
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )


            state = new_state       #Actualiza el estado para el siguiente paso
            state_p = new_state_p   #Actualiza los segmentos del estado actual
            state_v = new_state_v


            rewards+=reward         #Actualiza la recompensa (recompensa o castigo)
        #Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards
        #print('episode number: ', i,'reward: ', reward)

    env.close()

    # Save Q table to file
    if is_training:
        f = open('mountain_car_opt.pkl','wb')
        pickle.dump(q, f)
        f.close()
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car_opt.png')




if __name__ == '__main__':
    run(1000, is_training=True, render=False)     #Exploración

    #run(10, is_training=False, render=True)     #Explotación
