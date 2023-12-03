import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (512, 512)
IMAGE_SIZE = (256, 256)
GRID_SIZE = 3
PIECE_SIZE = IMAGE_SIZE[0] // GRID_SIZE
WHITE = (255, 255, 255)

# Load the image
image = pygame.image.load('Image.jpg')

# Resize the image to fit the specified size
image = pygame.transform.scale(image, IMAGE_SIZE)

# Create a Pygame window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Scattered Image')

# Create a list to hold the image pieces
pieces = []

# Cut the image into 9 equal parts
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x = j * PIECE_SIZE
        y = i * PIECE_SIZE
        piece = image.subsurface(pygame.Rect(x, y, PIECE_SIZE, PIECE_SIZE))
        pieces.append(piece)

# Shuffle the pieces
random.shuffle(pieces)

# Scatter the pieces randomly within the window
for piece in pieces:
    x = random.randint(0, WINDOW_SIZE[0] - PIECE_SIZE)
    y = random.randint(0, WINDOW_SIZE[1] - PIECE_SIZE)
    screen.blit(piece, (x, y))

# Update the display
pygame.display.flip()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
sys.exit()
