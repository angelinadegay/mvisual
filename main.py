import pygame
from audio import AudioProcessor
from visual import Visualizer
import config
import numpy as np

def main():
    pygame.init()
    screen = pygame.display.set_mode(config.SCREEN_SIZE)
    pygame.display.set_caption('Interactive Music Visualization')

    audio_processor = AudioProcessor()
    visualizer = Visualizer(screen)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Inside the main loop
        predictions = audio_processor.predict()
        if predictions is not None and not np.allclose(predictions.mean(axis=0), 0, atol=1e-3):
            visualizer.draw_visuals(predictions.mean(axis=0))
        else:
            visualizer.clear_visuals()  # No predictions or silence detected, clear the visuals

        pygame.display.flip()
        clock.tick(10)  # Limit the frame rate to 10 frames per second

    pygame.quit()

if __name__ == '__main__':
    main()
