# pyImageObjectIdentify
Script that using machine learning to identify objects in an image

Install packages
pip install -r requirements.txt

/your_project_folder
  └── objectDetection.py
  └── requirements.txt

To test
Download a pre-trained model like ssd_mobilenet_v2_coco from TensorFlow's model zoo.
Place an image file in your project directory and update the image_path.
Run objectDetection.py.
This example uses a pre-trained TensorFlow model to identify objects in an image. 
The script is efficient, easy to read, and follows best practices like function 
documentation. It should give you a list of detected object IDs that you can map to 
actual object names based on the pre-trained model's documentation.
