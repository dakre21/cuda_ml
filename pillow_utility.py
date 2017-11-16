from PIL import Image
import numpy
import glob
import os

# Get all jpg files
jpg = os.getcwd() + "/images"
jpg_files = glob.glob(jpg + "/*.jpg")

# Convert jpg files to greyscale
for jpg_file in jpg_files:
    # Load and write 8-bit greyscale image to file
    with Image.open(jpg_file, 'r').convert('L') as img:
        raw_file = os.path.splitext(jpg_file)[0] + '.raw'
        with open(raw_file, 'w') as f:
            data = list(img.getdata())
            str_data = ""
            for i in range(len(data)):
                str_data += str(data[i]) + " "

            f.write(str_data)
