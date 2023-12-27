# AI drone conversational system
tello2.py is responsible for controlling the flight of the drone, including the transmission and reception of network packets.

tello_army.py is the main program and creates a front-end interface through Pygame, allowing users to set paths, control the flight of the drone through tello2.py, and obtain real-time image streaming data. Through the yolov7 object detection model, once a suspicious object is detected, the system captures the photo and calls the function in blip2.py for image analysis. The obtained text results are converted into speech through the text-to-speech module. During program execution, the microphone will be turned on and the speech-to-text module will be used to listen to the user's instructions. Once any instructions are captured, a function in blip2.py is called, allowing the user to ask the AI ​​for details about the photo.

blip2.py is used to process conversational AI models that describe details of input images.

# Environment
The following packages need to be installed.

<pre><code>pip3 install torch==2.1.0 torchvision torchaudio
pip3 install transformers==4.34.1 bitsandbytes==0.41.1 peft==0.6.0 datasets scipy
pip3 install tk pygame numpy opencv-python
</code></pre>

# Run
Make sure that the computer and the drone are in the same local network, and the drone's IP is set to 192.168.10.1.

<pre><code>python3 tello_army.py</code></pre>
