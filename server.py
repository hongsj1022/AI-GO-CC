from __future__ import division

from flask import Flask, request, app, render_template, Response, redirect, url_for, send_from_directory

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

import argparse
import os
import logging
import cv2
import time
import datetime
import json
import imutils
import shutil
import paramiko
from scp import SCPClient, SCPException
app = Flask(__name__)

#Define logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logging.Formatter.converter = time.localtime
logger.addHandler(stream_handler)

#Define SSH Class
class SSHManager:
    """ usage:
        >>> import SSHManager
        >>> ssh_manager = SSHManager()
        >>> ssh_manager.create_ssh_client(hostname, username, password)
        >>> ssh_manager.send_command("ls -al")
        >>> ssh_manager.send_file("/path/to/local_path", "/path/to/remote_path")
        >>> ssh_manager.get_file("/path/to/remote_path", "/path/to/local_path")
        ...
        >>> ssh_manager.close_ssh_client()
    """
    def __init__(self):
        self.ssh_client = None

    def create_ssh_client(self, hostname, username, password):
        """Create SSH client session to remote server"""
        if self.ssh_client is None:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(hostname, username=username, password=password)
        else: print("SSH client session exist.")

    def close_ssh_client(self):
        """Close SSH client session"""
        self.ssh_client.close()

    def send_file(self, local_path, remote_path):
        """Send a single file to remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path, preserve_times=True)

        except SCPException:
            raise SCPException.message

    def get_file(self, remote_path, local_path):
        """Get a single file from remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)

        except SCPException:
            raise SCPException.message

    def send_command(self, command):
        """Send a single command"""
        stdin, stdout, stderr = self.ssh_client.exec_command(command)
        return stdout.readlines()


@app.route('/')
def index():

    return "Welcome to Training Server on Core!!"

@app.route('/tl_training', methods = ['GET','POST'])
def tl_training():

    logger.info("Start re-training trafficlight detection model.")

    os.system("unzip tl_train/images/images.zip -d /home2/icns/aigo/tl_train/images/")
    logger.info("Finish to unzip received tl_images.")
    os.system("rm /home2/icns/aigo/tl_train/images/images.zip")

    os.system("unzip tl_train/labels/labels.zip -d /home2/icns/aigo/tl_train/labels/")
    logger.info("Finish to unzip received tl_labels.")
    os.system("rm /home2/icns/aigo/tl_train/labels/labels.zip")

    remote_path = "/home/aigo/detect/trafficlight"
    
    # tl training
    os.system("python3 /home2/icns/aigo/tl/train.py")

    remote_file = remote_path + "/test4best.pt"
    local_file = "/home2/icns/aigo/tl_train/test4best.pt"
    ssh_manager.send_file(local_file, remote_path)
    ssh_manager.get_file(remote_file, local_file)
    logger.info("Success updated tl_model transmission.")
    os.system("rm /home2/icns/aigo/tl_train/images/*")
    os.system("rm /home2/icns/aigo/tl_train/labels/*")
    logger.info("Images and Labels removed.")

    return "TrafficLight Detection model has been updated!"


@app.route('/car_training', methods = ['GET','POST'])
def car_training():

    logger.info("Start re-training car detection and distance estimation model.")

    os.system("unzip car_train/images/images.zipi -d /home2/icns/aigo/car_train/images/")
    logger.info("Finish to unzip received car_images.")
    os.system("rm /home2/icns/aigo/car_train/images/images.zip")

    os.system("unzip car_train/labels/labels.zip -d /home2/icns/aigo/car_train/labels/")
    logger.info("Finish to unzip received car_labels.")
    os.system("rm /home2/icns/aigo/car_train/labels/labels.zip")

    remote_path = "/home/aigo/detect/car/weights"
    
    # car detection training
    os.system("python3 /home2/icns/aigo/car/train_detection.py"))

    time.sleep(1)

    # distance training
    os.system("python3 /home2/icns/aigo/car/train_distance.py"))

    remote_file_car = remote_path + "/tiny1_8000.pth"
    remote_file_dist = remote_path + "/distance_model.pth"
    local_file_car = "/home2/icns/aigo/car_train/tiny1_8000.pth")
    local_file_dist = "/home2/icns/aigo/car_train/distance_model.pth")
    ssh_manager.send_file(local_file_car, remote_path)
    ssh_manager.send_file(local_file_dist, remote_path)
    ssh_manager.get_file(remote_file_car, local_file_car)
    ssh_manager.get_file(remote_file_dist, local_file_car)
    logger.info("Success updated car_model transmission.")
    os.system("rm /home2/icns/aigo/car_train/images/*")
    os.system("rm /home2/icns/aigo/car_train/labels/*")
    logger.info("Images and Labels removed.")

    return "Car Detection model and Distance Estimation model have been updated!"


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    remote_server = "210.114.91.98"
    remote_id = "icns"
    remote_pw = "iloveicns"

    ssh_manager = SSHManager()
    ssh_manager.create_ssh_client(remote_server, remote_id, remote_pw)

    app.run(host = '0.0.0.0', port=10222)
