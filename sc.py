import os
import time

from pynput import mouse, keyboard
from PIL import ImageGrab
import threading

# Ensure the directory exists
output_dir = 'RawImages'
os.makedirs(output_dir, exist_ok=True)
press = 600
taking_screenshots = False

def take_screenshot():
    global press
    screenshot = ImageGrab.grab()
    filename = os.path.join(output_dir, f'image{press}.jpg')
    press += 1
    screenshot.save(filename)
    print(f'Screenshot saved: {press}')

def on_click(x, y, button, pressed):
    global press
    if taking_screenshots and button == mouse.Button.left and not pressed:
        screenshot_thread = threading.Thread(target=take_screenshot)
        screenshot_thread.start()

# Set up the listener
def on_press(key):
    global taking_screenshots
    try:
        if key.char == '[':
            taking_screenshots = True
            print('Started taking screenshots')
        elif key.char == ']':
            taking_screenshots = False
            print('Stopped taking screenshots')
    except AttributeError:
        pass

# Set up the listeners
mouse_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_press)

print('Listening for key presses and mouse clicks...')
keyboard_listener.start()
mouse_listener.start()
keyboard_listener.join()
mouse_listener.join()
