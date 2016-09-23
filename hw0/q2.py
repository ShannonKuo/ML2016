import sys
import Image
im = Image.open(sys.argv[1])
im = im.transpose(Image.ROTATE_180)
im.save("ans2.png")
