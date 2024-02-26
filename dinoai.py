import pygame
import sys
import random
import numpy as np
import pickle  # Aggiunto per la serializzazione
import matplotlib.pyplot as plt
import os
import math

def printlogo():
    print("░▒▓███████▓▒░  ░▒▓██████▓▒░  ░▒▓██████▓▒░ ░▒▓███████▓▒░ ░▒▓█▓▒░      ░▒▓████████▓▒░ ")
    print("░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░      ░▒▓█▓▒░        ")
    print("░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░      ░▒▓█▓▒░        ")
    print("░▒▓███████▓▒░░ ▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒ ░▒▓███████▓▒░ ░▒▓█▓▒░      ░▒▓██████▓▒░   ")
    print("░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░      ░▒▓█▓▒░        ")
    print("░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░      ░▒▓█▓▒░        ")
    print("░▒▓█▓▒░░▒▓█▓▒░ ░▒▓██████▓▒░  ░▒▓██████▓▒░ ░▒▓███████▓▒░ ░▒▓████████▓▒░▒▓████████▓▒░ ")
    print("")
    print(" ░▒▓███████▓▒░ ▒▓████████▓▒ ░▒▓█▓▒░░▒▓█▓▒ ░▒▓███████▓▒░░ ▒▓█▓▒░ ░▒▓██████▓▒░  ")
    print("░▒▓█▓▒░          ░▒▓█▓▒░    ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ")
    print("░▒▓█▓▒░          ░▒▓█▓▒░    ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ")
    print(" ░▒▓██████▓▒░    ░▒▓█▓▒░    ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ")
    print("       ░▒▓█▓▒░   ░▒▓█▓▒░    ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ")
    print("       ░▒▓█▓▒░   ░▒▓█▓▒░    ░▒▓█▓▒░░▒▓█▓▒ ░▒▓█▓▒░░▒▓█▓▒░ ▒▓█▓▒░ ▒▓█▓▒░░▒▓█▓▒░ ")
    print("░▒▓███████▓▒░    ░▒▓█▓▒░     ░▒▓██████▓▒░ ░▒▓███████▓▒░░ ▒▓█▓▒░ ░▒▓██████▓▒░  ")

os.environ['SDL_VIDEO_WINDOW_POS'] = '950,100'  # Sostituisci con le coordinate desiderate

# Inizializzazione di Pygame
pygame.init()

# Impostazioni del gioco
WIDTH, HEIGHT = 800, 300
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game")


class DinoGame:
    def __init__(self):
        self.dino = pygame.Rect(50, HEIGHT - 50, 50, 50)
        self.obstacles = []
        self.clock = pygame.time.Clock()
        self.spawn_frequency = 50
        self.spawn_counter = 0
        self.obstacle_speed = 5
        self.obstacle_gap = 200
        self.jump_speed = 10
        self.jump_velocity = 0

    printlogo()

    def reset(self):
        self.dino = pygame.Rect(50, HEIGHT - 50, 50, 50)
        self.obstacles = []

    def raycast(self):
        num_rays = 10
        ray_length = 200
        angles = [math.pi / 6 * i for i in range(-num_rays // 2, num_rays // 2 + 1)]

        # Lista per memorizzare le distanze agli ostacoli per ogni raggio
        distances = []

        for angle in angles:
            # Calcola la direzione del raggio
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)

            # Posizione iniziale del raggio
            ray_x = self.dino.x
            ray_y = self.dino.y

            # Esegui il raycasting
            for _ in range(ray_length):
                ray_x += ray_dx
                ray_y += ray_dy

                # Controlla se il raggio colpisce un ostacolo
                for obstacle in self.obstacles:
                    if obstacle.collidepoint(ray_x, ray_y):
                        # Calcola la distanza dal dinosauro all'ostacolo
                        distance = math.sqrt((ray_x - self.dino.x)**2 + (ray_y - self.dino.y)**2)
                        distances.append(distance)
                        break  # Esci dal ciclo interno se il raggio colpisce un ostacolo
                else:
                    continue  # Continua con il prossimo passo se il raggio non colpisce un ostacolo
                break  # Esci dal ciclo esterno se il raggio colpisce un ostacolo

            # Se il raggio non ha colpito un ostacolo, aggiungi la lunghezza massima al posto della distanza
            else:
                distances.append(ray_length)

        return distances

    def get_state(self):
        dino_speed = self.jump_velocity  # Aggiungi la velocità del dinosauro allo stato
        ray_distances = self.raycast()  # Aggiungi le distanze dei raggi come parte dello stato
        if self.obstacles:
            next_obstacle = self.obstacles[0]
            obstacle_distance_x = next_obstacle.x - self.dino.x
            obstacle_distance_y = self.dino.y - next_obstacle.height  # Distanza verticale dal prossimo ostacolo
            obstacle_height = next_obstacle.height
            num_obstacles = len(self.obstacles)
        else:
            obstacle_distance_x = WIDTH  # Se non ci sono ostacoli, impostiamo una distanza massima
            obstacle_distance_y = HEIGHT  # Allo stesso modo, l'altezza massima
            obstacle_height = 0  # Altezza dell'ostacolo impostata a 0
            num_obstacles = 0

        # Restituisci lo stato come un array concatenato di tutte le informazioni
        return np.concatenate(([self.dino.x, self.dino.y, dino_speed, obstacle_distance_x, obstacle_distance_y, obstacle_height, num_obstacles], ray_distances))
    
    def perform_action(self, action):
        # Esegui l'azione dell'agente (saltare)
        if action == 1 and self.dino.y == HEIGHT - 50:
            self.jump_velocity = -self.jump_speed  # Imposta la velocità di salto

    def update(self):
        # Movimento degli ostacoli
        for obstacle in self.obstacles:
            obstacle.x -= self.obstacle_speed

            # Rimuovi gli ostacoli che sono usciti dallo schermo
            if obstacle.right < 0:
                self.obstacles.remove(obstacle)

        # Spawn di nuovi ostacoli casualmente
        self.spawn_counter += 1
        if self.spawn_counter == self.spawn_frequency:
            obstacle = pygame.Rect(WIDTH, HEIGHT - 50, 20, 30)
            self.obstacles.append(obstacle)
            self.spawn_counter = 0

        # Simulazione del movimento verticale
        if self.dino.y < HEIGHT - 50 or self.jump_velocity < 0:
            self.dino.y += self.jump_velocity
            self.jump_velocity += 0.5  # Simula la gravità durante il salto

            if self.dino.y >= HEIGHT - 50:
                self.dino.y = HEIGHT - 50
                self.jump_velocity = 0  # Il dinosauro ha toccato il suolo

        # Verifica collisione con gli ostacoli
        for obstacle in self.obstacles:
            if self.dino.colliderect(obstacle):
                return True  # Collisione, partita terminata

        return False  # Nessuna collisione

    def draw_rays(self, screen):
        num_rays = 3
        ray_length = 350
        angle_increment = math.pi / (num_rays - 1)

        # Posizione iniziale del raggio (parte anteriore del dinosauro)
        dino_front_x = self.dino.x + self.dino.width
        dino_front_y = self.dino.y + self.dino.height // 2

        for i in range(num_rays):
            # Calcola l'angolo per il raggio corrente
            angle = angle_increment * i

            # Calcola la direzione del raggio
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)

            # Posizione iniziale del raggio
            ray_x = dino_front_x
            ray_y = dino_front_y

            # Esegui il raycasting
            for _ in range(ray_length):
                ray_x += ray_dx
                ray_y += ray_dy

                # Disegna il raggio
                pygame.draw.line(screen, (150, 150, 150), (dino_front_x, dino_front_y), (int(ray_x), int(ray_y)))


                # Controlla se il raggio colpisce un ostacolo
                for obstacle in self.obstacles:
                    if obstacle.colliderect(pygame.Rect(ray_x, ray_y, 1, 1)):
                        # Se l'ostacolo colpito è un cactus, interrompi il ciclo
                        if obstacle.height < HEIGHT:
                            break
                else:
                    continue  # Continua con il prossimo passo se il raggio non colpisce un ostacolo
                break  # Esci dal ciclo esterno se il raggio colpisce un ostacolo

    def render(self, episode, total_reward, epsilon):
        # Disegna gli elementi del gioco
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, self.dino)

        for obstacle in self.obstacles:
            pygame.draw.rect(screen, BLACK, obstacle)

        # Disegna i raggi del raycasting
        self.draw_rays(screen)

        # Mostra il punteggio corrente, l'episodio e il tasso di esplorazione
        font = pygame.font.Font(None, 36)
        score_text = font.render("Punteggio: {}".format(total_reward), True, BLACK)
        episode_text = font.render("Episodio: {}".format(episode), True, BLACK)
        epsilon_text = font.render("Esplorazione: {:.2}".format(epsilon), True, BLACK)

        screen.blit(score_text, (10, 10))
        screen.blit(episode_text, (10, 40))
        screen.blit(epsilon_text, (10, 70))

        pygame.display.flip()
        self.clock.tick(FPS)


# Agente Q-learning
class DinoAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.gamma = 0.95  # Factor of future rewards
        self.epsilon = 1.5  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.10

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Creazione del gioco e dell'agente
game = DinoGame()
state_size = 250000  # x del dino, y del dino, x dell'ostacolo
action_size = 2  # 0: non fare nulla, 1: saltare
agent = DinoAgent(state_size, action_size)


# Carica la Q-table da un file se presente
try:
    with open('q_table.pkl', 'rb') as file:
        # Verifica se il file è vuoto
        if file.readable():
            agent.q_table = pickle.load(file)
except (FileNotFoundError, EOFError):
    # Crea una nuova Q-table se il file non esiste o è vuoto
    agent.q_table = np.zeros((state_size, action_size))


avg_scores = []
episode_list = []

plt.ion()
fig, ax = plt.subplots()


# Ciclo principale di apprendimento
for episode in range(2000):
    game.reset()
    state = 0
    total_reward = 0

    for time in range(500):
        action = agent.act(state)
        game.perform_action(action)

        next_state = 0
        reward = 1 if not game.update() else -1
        done = True if reward == -1 else False

        total_reward += reward

        if total_reward == 189:
            # Punizione severa alla fine dell'episodio
            agent.update_q_table(state, action, -50, next_state, done)

            # Aumenta temporaneamente il tasso di esplorazione
            agent.epsilon *= 1.0 # Modifica il fattore di aumento secondo le tue esigenze
        else:
            # Aggiorna la Q-table normalmente con la ricompensa ottenuta
            agent.update_q_table(state, action, reward, next_state, done)

        game.render(episode, total_reward, agent.epsilon)
        pygame.time.delay(10)

        if done:
            print("Episodio {}: Punteggio: {}, Esplorazione: {:.2}".format(episode, total_reward, agent.epsilon))
            break

        agent.decay_epsilon()

    # Riduci il tasso di esplorazione dopo ogni episodio
    agent.decay_epsilon()

    avg_scores.append(total_reward)
    episode_list.append(episode)

    # Visualizza il grafico dopo ogni episodio
    ax.clear()
    ax.plot(episode_list, avg_scores, label='Punteggio Medio per Episodio')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Punteggio Medio')
    ax.set_title('Andamento del Punteggio Medio per Episodio')
    ax.legend()
    plt.draw()
    plt.pause(0.001)

    # Salva la Q-table dopo ogni episodio
    with open('q_table.pkl', 'wb') as file:
        pickle.dump(agent.q_table, file)





# Mantieni aperta la finestra del grafico alla fine dell'addestramento
plt.ioff()
plt.show()

pygame.quit()
sys.exit()

