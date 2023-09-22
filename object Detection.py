import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image file and prepare it for object detection.
    :param image_path: Path to the image file
    :return: Processed image tensor
    """
    image = Image.open(image_path)
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.array([input_array])
    return input_array

def detect_objects(model, image):
    """
    Perform object detection on the given image.
    :param model: Loaded TensorFlow model
    :param image: Image tensor
    :return: List of detected object labels
    """
    predictions = model.predict(image)
    class_ids = predictions['detection_classes'][0].astype(int)
    return class_ids

if __name__ == "__main__":
    # Download a pre-trained model (for example, MobileNet)
    model = tf.saved_model.load("ssd_mobilenet_v2_coco/saved_model")

    # Load an image file (replace with your own image path)
    image_path = "path/to/your/image.jpg"
    image_tensor = load_image(image_path)

    # Perform object detection
    detected_object_ids = detect_objects(model, image_tensor)
    print("Detected object class IDs:", detected_object_ids)
