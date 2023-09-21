import tensorflow as tf
from PIL import Image
import numpy as np

from utils import normalize, create_mask

model_selection = False
if model_selection:
    sel_model = input("Select model coco_128_128 (c) or DUTS_512_512 (d512): ")
    if sel_model == "c":
        print("Selected coco_128_128, loading model")
        model = tf.keras.models.load_model("./cocoset_seg/coco_set_unet_semantic_segmentation_3_epochs_128_128.keras")
        input_size = (128, 128)
    elif sel_model == "d512":
        print("Selected DUTS_512_512, loading model")
        model = tf.keras.models.load_model("./DUTS_set_model/DUTS_set_unet_semantic_segmentation_512_512_+4_epochs.keras")
        input_size = (512, 512)
    else:
        print("Invalid model selected")
        exit()
else:
    model = tf.keras.models.load_model("./DUTS_set_model/DUTS_set_unet_semantic_segmentation_512_512_+4_epochs.keras")
    input_size = (512, 512)
try:
    while True:
        image_path = input("Enter image path: ")
        
        # Load image and get it's size
        try:
            image = tf.image.decode_image(open(image_path, "rb").read())
        except FileNotFoundError:
            print("File not found")
            continue
        w, h, _ = image.shape

        # Resize image to input_size and normalize it
        image = tf.image.resize(image, input_size)
        image = normalize(image, 1)[0]

        # Predict mask
        pred_img = create_mask(model.predict(tf.expand_dims(image, 0)))
        pred_img = tf.cast(pred_img, tf.float32)
        
        # remove background then resize it to original size
        back_removed =  image * pred_img
        back_removed = back_removed * 255
        back_removed = tf.keras.utils.img_to_array(back_removed).astype(np.uint8)
        back_transparent = Image.fromarray(back_removed)
        back_transparent = back_transparent.convert("RGBA")
        datas = back_transparent.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)

        back_transparent.putdata(newData)
        back_transparent = back_transparent.resize((h, w))

        # Save the image to a file
        im_ext = image_path.split(".")[-1]
        im_name = image_path.split(f".{im_ext}")[0] + "_back_removed.png"
        back_transparent.save(im_name, "PNG")
        print(f"Saved image to {im_name}")
except KeyboardInterrupt:
    print("Exiting...")
    exit()