import pygame
import random

# Initialisation de PyGame
pygame.init()

# Dimensions de la fenêtre et de la grille
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Directions de mouvement
DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}

# Initialisation de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Labyrinthe Aléatoire")

# Fonction pour dessiner les cellules
def draw_cell(x, y, color):
    pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Fonction pour dessiner les murs d'une cellule
def draw_walls(x, y, walls):
    if walls[0] == "1":  # Mur en haut
        pygame.draw.line(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, y * CELL_SIZE), 2)
    if walls[1] == "1":  # Mur en bas
        pygame.draw.line(screen, BLACK, (x * CELL_SIZE, (y + 1) * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 2)
    if walls[2] == "1":  # Mur à gauche
        pygame.draw.line(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE), (x * CELL_SIZE, (y + 1) * CELL_SIZE), 2)
    if walls[3] == "1":  # Mur à droite
        pygame.draw.line(screen, BLACK, ((x + 1) * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 2)

# Génération du labyrinthe avec l'algorithme du parcours en profondeur
def generate_maze():
    visited = [[False for _ in range(COLS)] for _ in range(ROWS)]
    stack = []

    # Départ au coin supérieur gauche
    current_cell = (0, 0)
    visited[0][0] = True
    maze = [["1111" for _ in range(COLS)] for _ in range(ROWS)]  # Murs (haut, bas, gauche, droite)

    while True:
        x, y = current_cell
        neighbors = []

        # Trouver les voisins non visités
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS and not visited[ny][nx]:
                neighbors.append((nx, ny, direction))

        if neighbors:
            # Choisir un voisin au hasard
            nx, ny, direction = random.choice(neighbors)
            visited[ny][nx] = True
            stack.append(current_cell)

            # Enlever les murs
            if direction == "UP":
                maze[y][x] = maze[y][x][:0] + "0" + maze[y][x][1:]
                maze[ny][nx] = maze[ny][nx][:1] + "0" + maze[ny][nx][2:]
            elif direction == "DOWN":
                maze[y][x] = maze[y][x][:1] + "0" + maze[y][x][2:]
                maze[ny][nx] = maze[ny][nx][:0] + "0" + maze[ny][nx][1:]
            elif direction == "LEFT":
                maze[y][x] = maze[y][x][:2] + "0" + maze[y][x][3:]
                maze[ny][nx] = maze[ny][nx][:3] + "0" + maze[ny][nx][4:]
            elif direction == "RIGHT":
                maze[y][x] = maze[y][x][:3] + "0" + maze[y][x][4:]
                maze[ny][nx] = maze[ny][nx][:2] + "0" + maze[ny][nx][3:]

            current_cell = (nx, ny)
        elif stack:
            current_cell = stack.pop()
        else:
            break

    return maze

# Boucle principale
def main():
    maze = generate_maze()
    clock = pygame.time.Clock()
    running = True

    # Position du joueur
    player_x, player_y = 0, 0

    # Position de la cible
    target_x, target_y = COLS - 1, ROWS - 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Déplacement du joueur
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and maze[player_y][player_x][0] == "0":
            player_y -= 1
        elif keys[pygame.K_DOWN] and maze[player_y][player_x][1] == "0":
            player_y += 1
        elif keys[pygame.K_LEFT] and maze[player_y][player_x][2] == "0":
            player_x -= 1
        elif keys[pygame.K_RIGHT] and maze[player_y][player_x][3] == "0":
            player_x += 1

        screen.fill(WHITE)

        # Dessiner le labyrinthe
        for y in range(ROWS):
            for x in range(COLS):
                draw_cell(x, y, WHITE)
                draw_walls(x, y, maze[y][x])

        # Dessiner la cible
        draw_cell(target_x, target_y, RED)

        # Dessiner le joueur
        draw_cell(player_x, player_y, BLUE)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
