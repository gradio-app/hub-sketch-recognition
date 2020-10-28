import gradio as gr
import tensorflow as tf
import urllib.request

# mlp_model = tf.keras.models.load_model(
#   "models/sketch_recognition/mlp.h5")
cnn_model = tf.keras.models.load_model(
  "models/sketch_recognition/cnn.h5")

labels = urllib.request.urlopen("https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt")
labels = labels.read()
labels = labels.decode('utf-8').split("\n")[:-1]


def predict(img):
  img = tf.math.divide(img, 255)
  preds = cnn_model.predict(img.reshape(-1, 28, 28, 1))[0]
  return {label: float(pred) for label, pred in zip(labels, preds)}

output = gr.outputs.Label(num_top_classes=3)

title="Sketch Recognition"
description="This Convolution Neural Network was trained on Google's " \
            "QuickDraw dataset with 345 classes. Try it by drawing a " \
            "lightbulb, radio, or anything you can think of!"
thumbnail="https://github.com/gradio-app/machine-learning-experiments/raw/master/lightbulb.png?raw=true"
gr.Interface(predict, "sketchpad", output, live=True, title=title, analytics_enabled=False,
             description=description, thumbnail=thumbnail, capture_session=True).launch()
