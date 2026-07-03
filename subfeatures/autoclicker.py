import pyautogui
import time
import winsound  # For Windows beep sound

TOTAL_CLICKS = 55
INTERVAL = 5  # seconds between clicks

print(f"Starting in 3 seconds... please switch to Roboflow window.")
time.sleep(3)

start_time = time.time()

for click_count in range(1, TOTAL_CLICKS + 1):
    #print(f"Click {click_count}/{TOTAL_CLICKS}")
    
    # Countdown timer
    for remaining in range(INTERVAL, 0, -1):
        print(f"Next click in {remaining} seconds", end='\r')
        time.sleep(1)
    
    pyautogui.press('right')  # Press Right Arrow

end_time = time.time()
total_time = end_time - start_time

print(f"\nFinished {TOTAL_CLICKS} clicks!")
print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Used to alert me when done
frequency = 1000  # Hertz
duration = 500    # milliseconds
winsound.Beep(frequency, duration)
