from tkinter.tix import Tree
from turtle import delay
from tello2 import Tello
from blip2 import Blip2
import cv2
import pygame
import numpy as np
import time
from threading import Thread
import subprocess
from gtts import gTTS
from pygame import mixer
import speech_recognition as sr

# Speed of the drone
S = 40
FPS = 120
W = 960
H = 500
CMERA_SOURSE = 0
UNIT_MOVE = 5


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class FrontEnd(object):
    def __init__(self):

        pygame.init()


        pygame.display.set_caption("drone controller")
        self.screen = pygame.display.set_mode([W, H])


        self.tello = Tello()

        # velocities -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.send_rc_control = False

        self.drawFrame = np.zeros((H, W, 3), np.uint8)
        self.initialDraw = False
        self.drawFrameW = np.zeros((H, W, 3), np.uint8)
        self.drawFrameW[:, :, :] = 255
        self.startDraw = False
        self.showControl = True
        self.flyReady = False
        self.cleanPush = False
        self.droneLot = np.array([10., 230.])
        self.flyVector = []
        self.flyUnitVector = np.array([1, 0])
        self.rotateLock = False
        self.stepRemain = 0
        self.controlScale = 50  # 50-999
        self.controlScalepush = 0
        self.pushButton = False
        self.pauseMove = True
        self.peopleDetect = 0

        self.arrowR = pygame.image.load("img/arrowR.png")
        self.arrowL = pygame.image.load("img/arrowL.png")
        self.arrowD = pygame.image.load("img/arrowD.png")
        self.arrowU = pygame.image.load("img/arrowU.png")
        self.arrowUU = pygame.image.load("img/arrowUU.png")
        self.arrowDD = pygame.image.load("img/arrowDD.png")
        self.rotateR = pygame.image.load("img/rotateR.png")
        self.rotateL = pygame.image.load("img/rotateL.png")

        self.arrowRrect = self.arrowR.get_rect(center=(830+60, 240+60))
        self.arrowLrect = self.arrowL.get_rect(center=(650+60, 240+60))
        self.arrowDrect = self.arrowD.get_rect(center=(740+60, 340+60))
        self.arrowUrect = self.arrowU.get_rect(center=(740+60, 140+60))
        self.arrowUUrect = self.arrowUU.get_rect(center=(100+60, 140+60))
        self.arrowDDrect = self.arrowDD.get_rect(center=(100+60, 340+60))
        self.rotateRrect = self.rotateR.get_rect(center=(190+60, 240+60))
        self.rotateLrect = self.rotateL.get_rect(center=(10+60, 240+60))

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
        mixer.init()

    def takeoff(self):
        self.tello.takeoff()
        self.send_rc_control = True

    def run(self):

        if(CMERA_SOURSE == 0):
            self.tello.connect()
            self.tello.set_speed(self.speed)

            self.tello.streamoff()
            self.tello.streamon()
            frame_read = self.tello.get_frame_read()

        elif(CMERA_SOURSE == 1):
            inVideo = cv2.VideoCapture(0)
            inVideo.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            inVideo.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not inVideo.isOpened():
                exit("Can't open default camera!")

        labelsPath = "data/coco.names"

        LABELS = open(labelsPath).read().strip().split("\n")

        should_stop = False

        config = "data/yolov7-tiny.cfg"
        model = "data/yolov7-tiny.weights"
        net = cv2.dnn.readNetFromDarknet(config, model)
        ln = net.getUnconnectedOutLayersNames()

        if(cv2.cuda.getCudaEnabledDeviceCount()):
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    Thread(target=self.update, args=(), daemon=True).start()

                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouseXY = pygame.mouse.get_pos()
                    print(mouseXY)
                    if(mouseXY[1] > 450):
                        if(mouseXY[0] < 210):
                            self.showControl = not self.showControl
                             r = sr.Recognizer()
                            with sr.Microphone() as source:
                                audio = r.listen(source)
                            soundtext = ""
                            try:
                                soundtext = r.recognize_google(audio)
                            except sr.UnknownValueError:
                                pass
                            except sr.RequestError as e:
                                pass
                            imgdetail=Blip2.gettext(frame, soundtext)
                            tts = gTTs(imgdetail)
                            tts.save("imgdetail.mp3")
                            mixer.music.load('imgdetail.mp3')
                            mixer.music.play(1)
                            
                        else:
                            self.controlScalepush = mouseXY[0]

                    elif (mouseXY[1] < 38):
                        if(mouseXY[0] <= 130):
                            self.flyReady = True
                        else:
                            self.cleanPush = True
                    elif (self.showControl == False):
                        self.startDraw = True
                        if self.initialDraw == False:
                            self.curMouse = mouseXY
                            self.droneLot = np.array(
                                mouseXY).astype(np.float64)
                            self.initialDraw = True
                    else:
                        self.pushButton = True
                        if (self.arrowRrect.collidepoint(event.pos)):
                            self.left_right_velocity = S
                        elif (self.arrowLrect.collidepoint(event.pos)):
                            self.left_right_velocity = -S
                        elif (self.arrowDrect.collidepoint(event.pos)):
                            self.for_back_velocity = -S
                        elif (self.arrowUrect.collidepoint(event.pos)):
                            self.for_back_velocity = S
                        elif (self.arrowUUrect.collidepoint(event.pos)):
                            self.up_down_velocity = S
                        elif (self.arrowDDrect.collidepoint(event.pos)):
                            self.up_down_velocity = -S
                        elif (self.rotateRrect.collidepoint(event.pos)):
                            self.yaw_velocity = S
                        elif (self.rotateLrect.collidepoint(event.pos)):
                            self.yaw_velocity = -S

                elif event.type == pygame.MOUSEBUTTONUP:
                    mouseXY = pygame.mouse.get_pos()
                    if (self.flyReady):
                        if(mouseXY[1] < 38 and mouseXY[0] <= 130):
                            if(self.send_rc_control):
                                if CMERA_SOURSE == 0:
                                    Thread(target=self.tello.land,
                                           args=(), daemon=True).start()
                                self.send_rc_control = False
                            else:
                                print("fly~")
                                self.pauseMove = True
                                if CMERA_SOURSE == 0:
                                    Thread(target=self.takeoff,
                                           args=(), daemon=True).start()
                                elif CMERA_SOURSE == 1:
                                    self.send_rc_control = True
                    elif(self.cleanPush):
                        if (mouseXY[1] < 38 and mouseXY[0] > 130):
                            print("clean~")
                            self.drawFrame[:, :, :] = 0
                            self.drawFrameW[:, :, :] = 255
                            self.flyUnitVector = np.array([1, 0])
                            self.stepRemain = 0
                            self.for_back_velocity = 0
                            self.left_right_velocity = 0
                            self.up_down_velocity = 0
                            self.yaw_velocity = 0
                            self.flyVector = []
                            self.initialDraw = False
                            self.pauseMove = True
                        elif mouseXY[1] > 38:
                            self.pauseMove = not self.pauseMove
                    elif(self.controlScalepush):
                        tmp = int((mouseXY[0]-self.controlScalepush)*.4)
                        self.controlScale = max(
                            min(999, tmp+self.controlScale), 50)
                    elif(self.startDraw):
                        self.flyVector.append(
                            np.array([mouseXY[0]-self.curMouse[0], mouseXY[1]-self.curMouse[1]]))
                        cv2.line(self.drawFrame, self.curMouse,
                                 pygame.mouse.get_pos(), (255, 0, 0), 3)
                        cv2.line(self.drawFrameW, self.curMouse,
                                 pygame.mouse.get_pos(), (255, 0, 0), 3)
                        self.curMouse = pygame.mouse.get_pos()
                    elif(self.pushButton):
                        if (self.arrowRrect.collidepoint(event.pos)):
                            self.left_right_velocity = 0
                        elif (self.arrowLrect.collidepoint(event.pos)):
                            self.left_right_velocity = 0
                        elif (self.arrowDrect.collidepoint(event.pos)):
                            self.for_back_velocity = 0
                        elif (self.arrowUrect.collidepoint(event.pos)):
                            self.for_back_velocity = 0
                        elif (self.arrowUUrect.collidepoint(event.pos)):
                            self.up_down_velocity = 0
                        elif (self.arrowDDrect.collidepoint(event.pos)):
                            self.up_down_velocity = 0
                        elif (self.rotateRrect.collidepoint(event.pos)):
                            self.yaw_velocity = 0
                        elif (self.rotateLrect.collidepoint(event.pos)):
                            self.yaw_velocity = 0

                    self.flyReady = False
                    self.cleanPush = False
                    self.startDraw = False
                    self.pushButton = False
                    self.controlScalepush = 0

            self.screen.fill([0, 0, 0])

            if CMERA_SOURSE == 0:
                if frame_read.stopped:
                    break
                frame = frame_read.frame
            elif CMERA_SOURSE == 1:
                ok, frame = inVideo.read()
                if not ok:
                    print('Failed to read frame')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (W, H))

            self.frame = frame.copy()

            blob = cv2.dnn.blobFromImage(
                self.frame, scalefactor=1/255., size=(416, 416), mean=(0, 0, 0), swapRB=False)


            net.setInput(blob)
            start = time.perf_counter()
            layerOutputs = net.forward(ln)
            end = time.perf_counter()

         
            boxes = []
            confidences = []
            classIDs = []


            for output in layerOutputs:

                for detection in output:

                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    
                    if confidence > 0.3:
                        
                        box = detection[0:4] * np.array([W, H, W, H])

                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences, score_threshold=.3, nms_threshold=.4)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    if LABELS[classIDs[i]] == 'person':
                        self.peopleDetect = time.perf_counter()

                    cv2.rectangle(frame, (x, y), (x+w, y+h),
                                  (0, 0, 255), 5)

                    text = "{}: {:.4f},a:{}".format(
                        LABELS[classIDs[i]], confidences[i], w*h)
                    textSize = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(
                        frame, (x, y-10), (x+textSize[0][0], y), (255, 255, 255), 10)
                    cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 2)

            if self.initialDraw:
                cv2.rectangle(frame, (self.droneLot-20).astype(np.int32),
                              (self.droneLot+20).astype(np.int32), (0, 255, 0), -1)
                cv2.line(frame, self.droneLot.astype(np.int32),
                         (self.droneLot+self.flyUnitVector*18).astype(np.int32), (0, 0, 255), 3)
            if self.startDraw:
                cv2.line(frame, self.curMouse,
                         pygame.mouse.get_pos(), (255, 0, 0), 5)

            if self.send_rc_control and self.flyVector and self.pauseMove == False:
                angleDegree = angle_between(
                    self.flyVector[0], self.flyUnitVector)/3.141592653589793*180
                resultUnit = unit_vector(self.flyVector[0])
                if(angleDegree > 0.0001 and self.rotateLock == False):
                    self.rotateLock = True
                    vC = np.cross(self.flyVector[0], self.flyUnitVector)
                    Thread(target=self.rotateDrane,
                           args=(vC < 0, angleDegree, resultUnit), daemon=True).start()

                elif(angleDegree > 0.0001):
                    cv2.line(frame, self.droneLot.astype(np.int32),
                             (self.droneLot+resultUnit*18).astype(np.int32), (0, 0, 150), 10)
                elif(self.stepRemain == 0):
                    if resultUnit[0] != 0:
                        self.stepRemain = self.flyVector[0][0]/resultUnit[0]
                    else:
                        self.stepRemain = self.flyVector[0][1]/resultUnit[1]
                    self.flyCounter = time.perf_counter()
                    self.flyLinetime = self.stepRemain/100.*self.controlScale/S
                    self.droneLotTmp = self.droneLot
                    self.for_back_velocity = S
                else:
                    self.flyFpstime = time.perf_counter()
                    self.droneLot = self.droneLotTmp+resultUnit * \
                        ((self.flyFpstime-self.flyCounter)
                         * 100*S/self.controlScale)
                    if(self.flyFpstime-self.flyCounter >= self.flyLinetime):
                        self.for_back_velocity = 0
                        self.droneLot = self.droneLotTmp+self.flyVector[0]
                        self.stepRemain = 0
                        self.flyVector.pop(0)

            # People detected
            if (time.perf_counter()-self.peopleDetect) < 1.5:
                cv2.putText(frame, "Person detected!", (168, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                imgdetail=Blip2.gettext(frame)
                tts = gTTs(imgdetail)
                tts.save("imgdetail.mp3")
                mixer.music.load('imgdetail.mp3')
                mixer.music.play(1)

            # battery +WIFI
            if CMERA_SOURSE == 0:
                WIFISNR = subprocess.check_output(
                    ["netsh", "wlan", "show", "interfaces"])
                f = WIFISNR.find(b'\x53\x69\x67\x6e\x61\x6c')
                WIFISNRvalue = "-1" if f == - \
                    1 else WIFISNR[f+25:f+28].decode("utf-8")
            elif CMERA_SOURSE == 1:
                WIFISNR = subprocess.check_output(
                    ["netsh", "wlan", "show", "network", "mode=Bssid"])
                f = WIFISNR.find(b'\x53\x69\x67\x6e\x61\x6c')
                WIFISNRvalue = "-1" if f == - \
                    1 else WIFISNR[f+21:f+24].decode("utf-8")

            if CMERA_SOURSE == 0:
                text = "Battery: {}%             Wi-Fi strength:{}".format(
                    self.tello.get_battery(), WIFISNRvalue)
            elif CMERA_SOURSE == 1:
                text = "Battery: {}%             Wi-Fi strength:{}".format(
                    87, WIFISNRvalue)

            cv2.putText(frame, text, (5, 500 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)

            # ready
            if self.send_rc_control:
                text = "fly:{:>3d}s".format(
                    99 if CMERA_SOURSE == 1 else self.tello.get_flight_time())
            else:
                text = "fly:N"
            cv2.rectangle(frame, (2, 2), (131, 39), (255, 255, 255), 5)
            cv2.rectangle(frame, (2, 2), (131, 39), (0, 0, 0), 2)
            cv2.putText(frame, text, (5, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Height
            if CMERA_SOURSE == 0:
                text = "H:{}cm,Baro:{:.2f}cm".format(
                    self.tello.get_height(), self.tello.get_state_field('baro'))
            elif CMERA_SOURSE == 1:
                text = "H:{}cm,Baro:{:.2f}cm".format(37, 100.52)
            cv2.putText(frame, text, (145, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2)

            # Location
            if CMERA_SOURSE == 0:
                text = "pitch:{},roll:{},yaw:{}".format(
                    self.tello.get_pitch(), self.tello.get_roll(), self.tello.get_yaw())
            elif CMERA_SOURSE == 1:
                text = "pitch:{},roll:{},yaw:{}".format(
                    0, -2, 11)
            cv2.putText(frame, text, (516, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2)
            # Scale
            cv2.line(frame, (720, 485),
                     (820, 485), (0, 0, 255), 5)
            if(self.controlScalepush):
                tmp = int((pygame.mouse.get_pos()[0]-self.controlScalepush)*.4)
                text = ":{:>3d}cm".format(
                    max(min(999, tmp+self.controlScale), 50))
            else:
                text = ":{:>3d}cm".format(self.controlScale)
            cv2.putText(frame, text, (830, 495),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            np.maximum(frame[:, :, 0], self.drawFrame[:,
                                                      :, 0], out=frame[:, :, 0])
            np.minimum(frame[:, :, 1:], self.drawFrameW[:,
                                                        :, 1:], out=frame[:, :, 1:])

            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            if self.showControl:
                self.screen.blits([(frame, (0, 0)), (self.arrowR, (830, 240)), (self.arrowL, (650, 240)), (self.arrowD, (740, 340)), (
                    self.arrowU, (740, 140)), (self.rotateR, (190, 240)), (self.rotateL, (10, 240)), (self.arrowDD, (100, 340)), (self.arrowUU, (100, 140))])
            else:
                self.screen.blits([(frame, (0, 0))])
            pygame.display.update()

            time.sleep(1 / FPS)

        self.tello.end()

    def keydown(self, key):
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_k:  #
            self.tello.emergency()

    def keyup(self, key):
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.send_rc_control and CMERA_SOURSE == 0 and self.rotateLock == False:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                                       self.up_down_velocity, self.yaw_velocity)

    def rotateDrane(self, d, degree, resultUnit):
        if self.send_rc_control and CMERA_SOURSE == 0:
            self.tello.send_rc_control(0, 0, 0, 0)
            if(d):
                self.tello.send_control_command("cw {}".format(int(degree)))
            else:
                self.tello.send_control_command("ccw {}".format(int(degree)))
        time.sleep(5)
        self.flyUnitVector = resultUnit
        self.rotateLock = False


def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()
