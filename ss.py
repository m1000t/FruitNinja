import os
import time
from pynput import mouse, keyboard
from PIL import ImageGrab
import threading

# Ensure the directory exists
output_dir = 'RawImages'
os.makedirs(output_dir, exist_ok=True)
press = 0
taking_screenshots = False
stop_screenshots = threading.Event()

def take_screenshot():
    global press
    screenshot = ImageGrab.grab()
    filename = os.path.join(output_dir, f'image{press}.jpg')
    press += 1
    screenshot.save(filename)
    print(f'Screenshot saved: {press}')
    time.sleep(1)

def on_click(x, y, button, pressed):
    global taking_screenshots
    if button == mouse.Button.left:
        if pressed:
            if taking_screenshots:
                print('Starting screenshot thread')
                stop_screenshots.clear()
                screenshot_thread = threading.Thread(target=take_screenshot)
                screenshot_thread.start()
        else:
            if taking_screenshots:
                print('Stopping screenshot thread')
                stop_screenshots.set()

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

# Set up the listeners[]
mouse_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_press)

print('Listening for key presses and mouse clicks...')[]
keyboard_listener.start()
mouse_listener.start()
keyboard_listener.join()
mouse_listener.join()