import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle



"""
La siguiente funcion tiene como objetivo convertir a todos los elementos
del arreglo x en una distribución probabilística cuya suma siempre será 1
las probabilidades de cada valor en el arreglo serán actualizados a medida
que el modelo explora
"""
def softmax(x):
    return np.exp(x)/sum(np.exp(x))



def run(episodes, is_training=True, render=False):
    #Inicializa el entorno MountainCar-v0 de Gymnasium
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Dividir la posición y velocidad en 20 segmentos utilizando np.linspace()
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Posición entre -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Velocidad entre -0.07 and 0.07


    """
    El siguiente agente utilizará el algoritmo del gradiente para elegir las acciones 
    que tomará en cada turno, las acciones son determinadas por una probabilidad
    y dicha probabilidad influye en proceso de explotacion como de exploracion que
    vaya a tener el agente respecto a su tabla q 
    """

    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # Inicializar la tabla q con: 20x20x3 array  (3 acciones posibles)
    else:
        f = open('mountain_car_grad.pkl', 'rb')
        q = pickle.load(f)
        f.close()



    #Parámetros 
    discount_factor_g = 0.9 
    alpha = 0.8
    rewards_per_episode = np.zeros(episodes)
    epsilon = 1       
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()   # random number generator
    rewards_per_episode = np.zeros(episodes)    


    for i in range(episodes):
        
        state = env.reset()[0]      # posición inicial, velocidad inicial siempre 0
        state_p = np.digitize(state[0], pos_space)      #np.digitize() nos da a qué segmento pertenece el valor 
        state_v = np.digitize(state[1], vel_space)      


        terminated = False       
        recompensas = []
        rewards=0
        H = np.zeros(env.action_space.n)
        pi = softmax(H)


        #Bucle para cada paso dentro de un episodio.
        #Dos condiciones que terminan el juego: Alcanzar el objetivo ó Llamada a más de 1000 acciones
        while(not terminated and rewards>-1000):  #penalty structure: we get -1 for every action that the car takes


            if is_training and rng.random() < epsilon:
                    #la accion es elegida en funcion de la probabilidad que haya sido actualizada
                    #al incio todas las acciones tienen la misma probabilidad
                 action = np.random.choice(env.action_space.n, p=pi)
            else:
                action = np.argmax(q[state_p, state_v, :])  #Explotación: selecciona la mejor acción basada en Q-table
                #action = np.random.choice(env.action_space.n, p=pi)

            new_state,reward,terminated,_,_ = env.step(action)  #Le pasamos la acción al entorno. El nuevo estado consiste de una posición y una velocidad
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                ##Actualiza la tabla Q con la nueva información obtenida
                q[state_p, state_v, action] = q[state_p, state_v, action] + alpha * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )
            """
            Ahora asignaremos una pregeferencia a cada accion H[j], dando como resutlado que existan
            distintas proababilidades que pueden llegar a ser llevadas a cabo, la preferencia es actualizada
            cada vez que termina un episodio, con la recompensa obtenida de haber ejecutado una
            determinada acción
            """
            recompensas.append(reward)
            recompensa_media = np.mean(recompensas)
                
            for j in range(env.action_space.n):
                if j == action:
                    #probabilidad actualizada para la accion tomada
                    H[j] += alpha * (reward - recompensa_media) * (1 - pi[j])
                else:
                    #probabilidad para el resto de acciones
                    H[j] -= alpha * (reward - recompensa_media) * pi[j]


          
            """La probabilidad es actualizada en general, en funcion de los resutlados
            obtenidos al ejecutar el codigo anterior, el cual aumentará o disminuirá
            la probabilidad de una determinada accion en el episodio actual
            """
            pi = softmax(H)  


            state = new_state      #posicion, velocidad, estado actualizados
            state_p = new_state_p   
            state_v = new_state_v


            rewards+=reward        
      
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards
        #print('episode number: ', i,'reward: ', reward)




    # Guardamos la tabla q, y también el gráfico de las recompensas
    if is_training:
        f = open('mountain_car_grad.pkl','wb')
        pickle.dump(q, f)
        f.close()

        mean_rewards = np.zeros(episodes)
        for t in range(episodes):
            mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(mean_rewards)
        plt.savefig(f'mountain_car_grad.png')




if __name__ == '__main__':
    run(1000, is_training=True, render=False)     #Exploración

    
    #run(10, is_training=False, render=True)     #Explotación
