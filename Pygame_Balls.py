import socket
import joblib
import numpy as np
import pandas as pd
import pygame
import random

# Load the trained classifier
model = joblib.load('eeg_classifier_model.pkl')

# Feature names used during training
feature_names = [
        'Delta1', 'Delta2', 'Delta3', 'Delta4', 'Delta5', 'Delta6', 'Delta7', 'Delta8',
        'Theta1', 'Theta2', 'Theta3', 'Theta4', 'Theta5', 'Theta6', 'Theta7', 'Theta8',
        'Alpha1', 'Alpha2', 'Alpha3', 'Alpha4', 'Alpha5', 'Alpha6', 'Alpha7', 'Alpha8',
        'BetaLow1', 'BetaLow2', 'BetaLow3', 'BetaLow4', 'BetaLow5', 'BetaLow6', 'BetaLow7', 'BetaLow8',
        'BetaMid1', 'BetaMid2', 'BetaMid3', 'BetaMid4', 'BetaMid5', 'BetaMid6', 'BetaMid7', 'BetaMid8',
        'BetaHigh1', 'BetaHigh2', 'BetaHigh3', 'BetaHigh4', 'BetaHigh5', 'BetaHigh6', 'BetaHigh7', 'BetaHigh8',
        'Gamma1', 'Gamma2', 'Gamma3', 'Gamma4', 'Gamma5', 'Gamma6', 'Gamma7', 'Gamma8',
        'Delta_Avg', 'Theta_Avg', 'Alpha_Avg', 'BetaLow_Avg', 'BetaMid_Avg', 'BetaHigh_Avg', 'Gamma_Avg',
        'Delta_Bipolar_Avg', 'Theta_Bipolar_Avg', 'Alpha_Bipolar_Avg', 'BetaLow_Bipolar_Avg', 'BetaMid_Bipolar_Avg', 'BetaHigh_Bipolar_Avg', 'Gamma_Bipolar_Avg'

    ]
# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EEG Classifier Visualization")

# Explicit color definitions
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
COLORS = {"Crunch": RED, "Blink": GREEN, "Relaxed": YELLOW}

# UDP Socket setup
UDP_IP = "127.0.0.1"
UDP_PORT = 1000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(0)  # Set socket to non-blocking mode

# Ball class
class Ball:
    def __init__(self, color, size, position):
        self.color = color
        self.size = size
        self.position = position

    def draw(self):
        pygame.draw.circle(screen, self.color, self.position, self.size)

# Main function
def main():
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill((0, 0, 0))  # Clear screen

        try:
            # Receive data from UDP
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            eeg_data_str = data.decode()
            eeg_data_list = eeg_data_str.split(',')

            if len(eeg_data_list) == 70:
                eeg_data = [float(x) if x != 'NaN' else 0.0 for x in eeg_data_list]
                eeg_df = pd.DataFrame([eeg_data], columns=feature_names)
                prediction = model.predict(eeg_df)
                
                # Create a ball based on the prediction
                color = COLORS.get(prediction[0], (255, 255, 255))  # Default to white if unknown
                size = random.randint(10, 50)
                position = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
                ball = Ball(color, size, position)
                ball.draw()

        except BlockingIOError:
            pass  # No data received, skip this iteration

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(60)  # 60 frames per second

    pygame.quit()

if __name__ == "__main__":
    main()
