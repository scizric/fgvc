from PIL import Image

def read_image(path):
    return Image.open(path)

def show_image(img):
    img.show()
    
def save_image(img,file_name):
    img.save(file_name)