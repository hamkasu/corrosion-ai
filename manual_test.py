from PIL import Image
img = Image.new("RGB", (100, 100), "red")
img.save("annotated/test_debug.jpg")