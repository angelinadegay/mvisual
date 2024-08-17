import pygame
import numpy as np

class Visualizer:
    def __init__(self, screen):
        self.screen = screen

    def draw_visuals(self, predictions):
        if predictions is None or len(predictions) == 0:
            print("No predictions to draw")
            return
        print("Drawing visuals with predictions:", predictions)
        self.screen.fill((0, 0, 0))  # Clear the screen

        num_bars = len(predictions)
        bar_width = self.screen.get_width() / num_bars

        for i, prediction in enumerate(predictions):
            bar_height = prediction * 300  # Scale the prediction value
            color = (255, int(prediction * 255), 255 - int(prediction * 255))  # Color based on prediction
            pygame.draw.rect(self.screen, color, (i * bar_width, self.screen.get_height() - bar_height, bar_width, bar_height))

        pygame.display.flip()
