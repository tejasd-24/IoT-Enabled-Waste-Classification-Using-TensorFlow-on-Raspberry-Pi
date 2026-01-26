import numpy as np
import cv2
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import open_stage

from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
print("Using tflite-runtime for inference.")

USD_PATH = "/home/jpst/Downloads/Waste_Project/robotic_arm.usdc"
MODEL_PATH = "/home/jpst/Downloads/waste_classifier_3_class_v14.tflite"

open_stage(USD_PATH)

world = World(stage_units_in_meters=1.0)

camera_prim = "/World/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
camera = Camera(
    prim_path=camera_prim,
    resolution=(224, 224),
    frequency=10
)
world.scene.add(camera)
world.reset()

interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

classes = ["Biodegradable_waste", "E_waste", "Non-Biodegradable_waste"]

print("\nWaste classification started (existing scene loaded)...")
print("--------------------------------------")

while simulation_app.is_running():
    world.step(render=True)

    frame = camera.get_rgba()
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) == 1:
        continue

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(rgb_image, (input_shape[2], input_shape[1]))
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0  # normalize

    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(output_data))
    pred_class = classes[pred_idx]
    confidence = float(output_data[pred_idx]) * 100.0

    print(f"Predicted Waste Type: {pred_class} ({confidence:.2f}%)")

simulation_app.close()
