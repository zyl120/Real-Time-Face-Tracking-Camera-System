# This Python file uses the following encoding: utf-8
import sys
import os

import cv2
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from multiprocessing import shared_memory, Process, Array
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo
from time import sleep
import signal
from queue import Queue
from dataclasses import dataclass, field
import subprocess

from ui_form import Ui_GUI
my_factory = PiGPIOFactory()
servo_0 = Servo(12, pin_factory=my_factory)
servo_1 = Servo(13, pin_factory=my_factory)

gui_pid = 0
face_pid = 0

@dataclass(order=True)
class ControlItem:
    """
    Used for notification queue to show notifications on taskbar
    """
    auto_control: bool
    servo_num: int
    value: int

def parent_signal_handler(signum, frame):
    print("INFO: {} received sig {}.".format(os.getpid(), signum))
    # Used as a single handler to close all child processes.
    if (signum == signal.SIGINT):
        os.kill(gui_pid, signal.SIGINT)
        os.waitpid(gui_pid, 0)
        os.kill(face_pid, signal.SIGINT)
        os.waitpid(face_pid, 0)
        print("INFO: other processes terminated")
        # close and unlike the shared memory
        # shm_block.close()
        # shm_block.unlink()
        print("INFO: shared memory destroyed")
        print("INFO: main process {} exited.".format(os.getpid()))
        sys.exit(0)

def child_signal_handler(signum, frame):
    # close child processes
    print("INFO: {} received sig {}.".format(os.getpid(), signum))
    if (signum == signal.SIGINT):
        print("INFO: child process {} exited.".format(os.getpid()))
        sys.exit(0)

def face_recog(shared_array):
    labels = ["Zilin", "Unknown"]

    face_cascade = cv2.CascadeClassifier('/home/pi/ECE-5725/finalProject/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("/home/pi/ECE-5725/finalProject/face-trainner.yml")

    cap = cv2.VideoCapture(0)
    print(cap)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(width)
    print(height)

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    
    
    while(True):
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # Recognize faces

        for (x, y, w, h) in faces:
            with shared_array.get_lock():
                shared_array[0] = x
                shared_array[1] = y
                shared_array[2] = w
                shared_array[3] = h
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            if conf >= 75:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                # cv2.putText(img, name, (x,y), font, 1, (0,0,255), 2)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        
        cv2.imshow("Preview", img)
        

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class Face_Result_Worker(QThread):
    motor_signal = pyqtSignal(float, float)
    def __init__(self, shared_array, control_queue):
        super(Face_Result_Worker, self).__init__()
        self.shared_array = shared_array
        self.control_queue = control_queue
        self.center_x = 0
        self.old_center_x = 0
        self.center_y = 0
        self.old_center_y = 0
        self.x_value = 0
        self.y_value = 0
        self.auto_control = True
    
    def run(self):
        while True:
            if(not self.control_queue.empty()):
                self.control_item = self.control_queue.get()
                self.auto_control = self.control_item.auto_control
                print(self.auto_control)
            
            if(self.auto_control):
                with self.shared_array.get_lock():
                    self.center_x = self.shared_array[0] + self.shared_array[2] / 2
                    self.center_y = self.shared_array[1] + self.shared_array[3] / 2
                if(self.old_center_x != self.center_x):
                    self.x_value += (2.0/640.0/10) * (320 - self.center_x)
                if(self.old_center_y != self.center_y):
                    self.y_value += (2.0/480.0/10) * (self.center_y - 240)
                self.old_center_x = self.center_x
                self.old_center_y = self.center_y
                if(self.x_value > 1):
                    self.x_value = 1
                if(self.x_value < -1):
                    self.x_value = -1
                if(self.y_value > 1):
                    self.y_value = 1
                if(self.y_value < -1):
                    self.y_value = -1
                servo_1.value = self.x_value
                servo_0.value = self.y_value
                sleep(0.15)
            else:
                if(self.control_item.servo_num == 0):
                    servo_0.value = self.control_item.value
                else:
                    servo_1.value = self.control_item.value
                sleep(0.15)
            self.motor_signal.emit(servo_1.value, servo_0.value)

            
    def stop(self):
        servo_0.detach()
        servo_1.detach()
        self.terminate()


class GUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_GUI()
        self.ui.setupUi(self)
        self.setFixedSize(350, 450)
        self.show()
        self.control_queue = Queue()
        self.ui.pushButton_Up.clicked.connect(self.go_up)
        self.ui.pushButton_Down.clicked.connect(self.go_down)
        self.ui.pushButton_Left.clicked.connect(self.go_left)
        self.ui.pushButton_Right.clicked.connect(self.go_right)
        self.ui.pushButton_Center.clicked.connect(self.go_center)
        self.ui.shutdownButton.clicked.connect(self.shutdown)
        self.ui.rebootButton.clicked.connect(self.reboot)
        
        self.ui.auto_button.toggled.connect(self.auto_control)
        self.ui.auto_button.setChecked(True)
        
        self.face_worker = Face_Result_Worker(shared_array, self.control_queue)
        self.face_worker.motor_signal.connect(self.change_display)
        self.x_value = 0
        self.y_value = 0
        
        self.face_worker.start()

    def shutdown(self):
        subprocess.run(["sudo", "shutdown", "-h", "now"]) 
    
    def reboot(self):
        subprocess.run(["sudo", "reboot"]) 


    def auto_control(self):
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 0,0))

    def go_up(self):
        self.y_value -= 0.1
        if(self.y_value < -1):
            self.y_value = -1
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 0,self.y_value))
        print("go up")
    
    def go_down(self):
        self.y_value += 0.1
        if(self.y_value > 1):
            self.y_value = 1
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 0,self.y_value))
        print("go down")
    
    def go_left(self):
        self.x_value -= 0.1
        if(self.x_value < -1):
            self.x_value = -1
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 1,self.x_value))
        print("go left")
    
    def go_right(self):
        self.x_value += 0.1
        if(self.x_value > 1):
            self.x_value = 1
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 1,self.x_value))
        print("go right")
    
    def go_center(self):
        self.x_value = 0
        self.y_value = 0
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 0,0))
        self.control_queue.put(ControlItem(self.ui.auto_button.isChecked(), 1,0))

    def change_display(self, x, y):
        self.ui.motorPosLabel.setText("Motor X: {}, Y: {}".format(x, y))


def fork_gui(shared_array):
    pid = os.fork()
    if (pid > 0): # parent process
        print("INFO: gui_pid={}".format(pid))
        return pid
    else:
        signal.signal(signal.SIGINT, child_signal_handler)
        os.environ["GPIOZERO_PIN_FACTORY"] = "pigpio"
        app = QApplication(sys.argv)
        widget = GUI()
        
        sys.exit(app.exec())

def fork_face(shared_array):
    pid = os.fork()
    if (pid > 0): # parent process
        print("INFO: face_pid={}".format(pid))
        return pid
    else:
        signal.signal(signal.SIGINT, child_signal_handler)
        face_recog(shared_array)

if __name__ == "__main__":
    
    shared_array = Array("i", (0, 0, 0, 0))
    gui_pid = fork_gui(shared_array)
    face_pid = fork_face(shared_array)
    print(gui_pid)
    print(face_pid)
    signal.signal(signal.SIGINT, parent_signal_handler)
