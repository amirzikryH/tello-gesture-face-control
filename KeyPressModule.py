import pygame

# Initialize Pygame and create a window
def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Drone Controls")

# Function to check if a specific key is pressed
def getKey(keyName):
    ans = False
    # Process events to keep window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    
    # Check if the key is pressed
    if keyInput[myKey]:
        ans = True
    
    pygame.display.update()
    return ans

# Main loop to test key presses
def main():
    if getKey("LEFT"):
        print("Left key pressed")
    if getKey("RIGHT"):
        print("Right key Pressed")
    if getKey("e"):
        print("Takeoff key pressed")
    if getKey("q"):
        print("Land key pressed")

# Run the initialization and main loop
if __name__ == "__main__":
    init()
    while True:
        main()