import pygame
from audio import AudioProcessor
from visual import Visualizer
import config

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

        predictions = audio_processor.predict()
        if predictions is not None:
            print(f"Predictions Mean: {predictions.mean(axis=0)}")  # Debugging print statement
            visualizer.draw_visuals(predictions.mean(axis=0))
        else:
            print("No predictions received")
        pygame.display.flip()
        clock.tick(10)  # Limit the frame rate to 10 frames per second

    pygame.quit()

if __name__ == '__main__':
    main()
