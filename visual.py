import pygame
import numpy as np

class Visualizer:
    def __init__(self, screen):
        self.screen = screen
        self.current_bars = None  # Track the current bar heights

    def draw_visuals(self, predictions):
        if predictions is None or len(predictions) == 0:
            self.clear_visuals()
            return

        print("Drawing visuals with predictions:", predictions)
        self.screen.fill((0, 0, 0))  # Clear the screen

        num_bars = len(predictions)
        bar_width = self.screen.get_width() / num_bars

        if self.current_bars is None:
            self.current_bars = np.zeros(num_bars)  # Initialize bars on the first run

        # Update current bars based on predictions with smoothing
        self.current_bars = 0.8 * self.current_bars + 0.2 * predictions * 300  # Smooth transition

        for i, bar_height in enumerate(self.current_bars):
            color = (255, int(self.current_bars[i] / 300 * 255), 255 - int(self.current_bars[i] / 300 * 255))  # Color based on prediction
            pygame.draw.rect(self.screen, color, (i * bar_width, self.screen.get_height() - bar_height, bar_width, bar_height))

        pygame.display.flip()

    def clear_visuals(self):
        print("Clearing visuals")
        self.screen.fill((0, 0, 0))  # Clear the screen
        self.current_bars = None  # Reset the bars
        pygame.display.flip()
