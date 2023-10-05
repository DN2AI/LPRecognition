# Webcam
from IPython.display import display, Javascript, Image
import PIL
import io
import numpy as np

from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode

import argparse
import time
from pathlib import Path

# Yolov7
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# LP Check
from check_LP import *

opt = {
    'weight': '/content/drive/MyDrive/Yolov7_Train/train_models/LP_detect_yolov7_500img.pt',
	'source': '0',
	'img_size': 640,
	'conf_thres': 0.25,
	'iou_thres': 0.45,
	'device': '',
	'view_img': False,
	'save_txt': False,
	'save_conf': False,
	'nosave': False,
	'classes': None,
	'agnostic_nms': False,
	'augment': False,
	'update': False,
	'project': 'runs/detect',
	'name': 'exp',
	'exist_ok': False,
	'no_trace': True,
}

class ColabWebCam:
    def __init__(self):
        print('Running...')

    def video_stream(self):
        js = Javascript('''
            var video;
            var div = null;
            var stream;
            var captureCanvas;
            var imgElement;
            var labelElement;
        
            var pendingResolve = null;
            var shutdown = false;
        
            function removeDom() {
                stream.getVideoTracks()[0].stop();
                video.remove();
                div.remove();
                video = null;
                div = null;
                stream = null;
                imgElement = null;
                captureCanvas = null;
                labelElement = null;
            }
        
            function onAnimationFrame() {
                if (!shutdown) {
                    window.requestAnimationFrame(onAnimationFrame);
                }
                if (pendingResolve) {
                    var result = "";
                    if (!shutdown) {
                    captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                    result = captureCanvas.toDataURL('image/jpeg', 0.8)
                    }
                    var lp = pendingResolve;
                    pendingResolve = null;
                    lp(result);
                }
            }
        
            async function createDom() {
                if (div !== null) {
                    return stream;
                }

                div = document.createElement('div');
                div.style.border = '2px solid black';
                div.style.padding = '3px';
                div.style.width = '100%';
                div.style.maxWidth = '600px';
                document.body.appendChild(div);
          
                const modelOut = document.createElement('div');
                modelOut.innerHTML = "Status:";
                labelElement = document.createElement('span');
                labelElement.innerText = 'No data';
                labelElement.style.fontWeight = 'bold';
                modelOut.appendChild(labelElement);
                div.appendChild(modelOut);
               
                video = document.createElement('video');
                video.style.display = 'block';
                video.width = div.clientWidth - 6;
                video.setAttribute('playsinline', '');
                video.onclick = () => { shutdown = true; };
                stream = await navigator.mediaDevices.getUserMedia(
                    {video: { facingMode: "environment"}});
                div.appendChild(video);

                imgElement = document.createElement('img');
                imgElement.style.position = 'absolute';
                imgElement.style.zIndex = 1;
                imgElement.opacity = 0.5
                imgElement.onclick = () => { shutdown = true; };
                div.appendChild(imgElement);
          
                const instruction = document.createElement('div');
                instruction.innerHTML = 
                    '' +
                    'Bấm vào video để dừng';
                div.appendChild(instruction);
                instruction.onclick = () => { shutdown = true; };
          
                video.srcObject = stream;
                await video.play();

                captureCanvas = document.createElement('canvas');
                captureCanvas.width = 640; //video.videoWidth;
                captureCanvas.height = 480; //video.videoHeight;
                window.requestAnimationFrame(onAnimationFrame);
          
                return stream;
            }
            async function stream_frame(label, imgData) {
                if (shutdown) {
                    removeDom();
                    shutdown = false;
                    return '';
                }

                var preCreate = Date.now();
                stream = await createDom();
          
                var preShow = Date.now();
                if (label != "") {
                    labelElement.innerHTML = label;
                }
                
                if (imgData != "") {
                    var videoRect = video.getClientRects()[0];
                    imgElement.style.top = videoRect.top + "px";
                    imgElement.style.left = videoRect.left + "px";
                    imgElement.style.width = videoRect.width + "px";
                    imgElement.style.height = videoRect.height + "px";
                    imgElement.src = imgData;
                }
          
                var preCapture = Date.now();
                var result = await new Promise(function(resolve, reject) {
                    pendingResolve = resolve;
                });
                shutdown = false;
          
                return {'create': preShow - preCreate, 
                    'show': preCapture - preShow, 
                    'capture': Date.now() - preCapture,
                    'img': result};
            }
        ''')

        display(js)
  
    def video_frame(self, label, bbox):
        data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
        return data

    # function to convert the JavaScript object into an OpenCV image
    def js_to_image(self, js_reply):
        """
        Params:
                js_reply: JavaScript object containing image from webcam
        Returns:
                img: OpenCV BGR image
        """
        # decode base64 image
        image_bytes = b64decode(js_reply.split(',')[1])
        # convert bytes to numpy array
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        # decode numpy array into OpenCV BGR image
        img = cv2.imdecode(jpg_as_np, flags=1)

        return img

    # function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
    def bbox_to_bytes(self, bbox_array):
        """
            Params:
                    bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
            Returns:
                    bytes: Base64 image byte string
        """
        # convert array into PIL image
        bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
        iobuf = io.BytesIO()
        # format bbox into png for return
        bbox_PIL.save(iobuf, format='png')
        # format return string
        bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

        return bbox_bytes
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 144, 144), auto=True, scaleFill=False, scaleup=True, stride=32):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
  
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
  
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
    def start_webcam(self):
        self.video_stream()
        label_html = 'Capturing...'
        bbox = ''

        with torch.no_grad():
            weights, imgsz = opt['weight'], (480, 640)
	        # Initialize
            set_logging()
            device = select_device('')
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            # imgsz = check_img_size(imgsz, s=stride)  # check img_size
    	
            if half:
                model.half()  # to FP16
        	
	        # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
       	        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
            
            while True:
              js_reply = self.video_frame(label_html, bbox)
              if not js_reply:
                break
        
              img0 = self.js_to_image(js_reply['img'])
              bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)
              img = self.letterbox(img0, imgsz, stride=stride)[0]
              img = img[:, :, ::-1].transpose(2, 0, 1)
              img = np.ascontiguousarray(img)
              img = torch.from_numpy(img).to(device)
              img = img.half() if half else img.float()  # uint8 to fp16/32
              img /= 255.0  # 0 - 255 to 0.0 - 1.0
              if img.ndimension() == 3:
                img = img.unsqueeze(0)
		    
	            # Inference
              t1 = time_synchronized()
              with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
              t2 = time_synchronized()

	            # Apply NMS
              pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
              t3 = time_synchronized()
		
	            # Process detections
              for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                  # Rescale boxes from img_size to im0 size
                  det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                  
                  if USE_SORT_TRACKER:
                    detections_ = []
                    
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                      detections_.append([x1, y1, x2, y2, conf])
                    
                    track_ids = sort_tracker.update(np.asarray(detections_))
		            # Draw boxes for visualization
                    for track in track_ids:
                      x1, y1, x2, y2, LP_id = track
                      license_plate_text = read_license_plate(img0, x1, y1, x2, y2)
                      assign_number_license_plate(LP_id, license_plate_text, img0[int(y1):int(y2), int(x1):int(x2), :])
                      label = f'ID: {int(LP_id)} {license_plate[LP_id]["text"]}'
                      bbox_array = plot_one_box([x1, y1, x2, y2], bbox_array, label=label, line_thickness=4)
                  else:
                      # Write results
                      for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        license_plate_text = read_license_plate(img0, x1, y1, x2, y2)
                        is_correct_format = check_format_number_license_plate(license_plate_text)
                        label = license_plate_text if is_correct_format else ''
                        bbox_array = plot_one_box([x1, y1, x2, y2], bbox_array, label=label, line_thickness=4)
      
                elif USE_SORT_TRACKER:
		                track_ids = sort_tracker.update() # SORT should be updated even with no detections
              
              bbox_array[:, :, 3] = (bbox_array.max(axis = 2) > 0).astype(int) * 255
              bbox_bytes = self.bbox_to_bytes(bbox_array)
              bbox = bbox_bytes

cam = ColabWebCam()
cam.start_webcam()
