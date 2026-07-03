import os

for file in os.listdir():
    if file.lower().endswith(".jpg"):
        new_name = file.replace("_png_jpg", "")
        if new_name != file:
            os.rename(file, new_name)
