## IoT-Enabled Waste Classification Using TensorFlow on Raspberry Pi

This project implements a **smart waste classification system** that identifies and categorizes waste into **biodegradable, non-biodegradable, and e-waste** using a **TensorFlow-based machine learning model** deployed on a **Raspberry Pi**. The system is designed as a prototype for **automated waste segregation** in smart waste management and environmental monitoring applications.

A camera module captures images of waste items, which are processed locally on the Raspberry Pi using a trained classification model. The classification results are published using an **MQTT-based publish–subscribe architecture**, enabling real-time data transmission to a central monitoring system.

In addition to waste classification, the system integrates **air quality sensors** to monitor environmental conditions around the waste management area, providing insights into pollution levels and overall plant health. The project emphasizes **edge-level inference**, **low-cost hardware**, and **scalable IoT communication**.

### Key Features
- Waste classification into **bio, non-bio, and e-waste** using TensorFlow  
- **On-device inference** on Raspberry Pi  
- **MQTT publish–subscribe** communication for real-time updates  
- Integrated **air quality monitoring** for environmental assessment  
- Modular design suitable for **smart city and industrial prototypes**

### Technologies Used
- Raspberry Pi  
- TensorFlow / TensorFlow Lite  
- Python  
- MQTT  
- Camera module and air quality sensors
- Isaac Sim
