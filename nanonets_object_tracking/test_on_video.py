import numpy as np
import cv2
import argparse
from deepsort import *
import pandas as pd


def get_gt(image,frame_id,gt_dict):

	if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
		return None,None,None

	frame_info = gt_dict[frame_id]

	detections = []
	out_scores = []
	for i in range(len(frame_info)):

		coords = frame_info[i]['coords']

		x1,y1,w,h = coords

		detections.append([x1,y1,w,h])
		out_scores.append(frame_info[i]['conf'])

	return detections,out_scores


def get_dict(filename):
	with open(filename) as f:	
		d = f.readlines()

	d = list(map(lambda x:x.strip(),d))

	last_frame = int(d[-1].split(',')[0])

	gt_dict = {x:[] for x in range(last_frame+1)}

	for i in range(len(d)):
		a = list(d[i].split(','))
		a = list(map(float,a))	

		coords = a[2:6]
		confidence = a[6]
		gt_dict[a[0]].append({'coords':coords,'conf':confidence})

	return gt_dict


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', help="path to MOT file")
	parser.add_argument('-v', '--video', help="Video to run deepsort on")
	parser.add_argument('-n', '--name', 
					 default="out.mot",
					 help="MOT csv name")
	args = parser.parse_args()
	csv_name = args.name

	#Load detections for the video. Options available: yolo,ssd and mask-rcnn
	filename = args.filename #'../pneuma-yolo7.mot'
	gt_dict = get_dict(filename)

	cap = cv2.VideoCapture(args.video)

	#Initialize deep sort.
	deepsort = deepsort_rbc()

	frame_id = 1

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('ssd_out_3.avi',fourcc, 10.0, (1920,1080))


	data = []
	while True:
		ret,frame = cap.read()
		if ret is False:
			frame_id+=1
			break	

		frame = frame.astype(np.uint8)

		detections,out_scores = get_gt(frame,frame_id,gt_dict)

		if detections is None:
			print("No dets")
			frame_id+=1
			continue

		detections = np.array(detections)
		out_scores = np.array(out_scores) 

		tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

		for track, det in zip(tracker.tracks, detections_class):
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
	
			data.append({
                "frame": frame_id, "id": track.track_id, 
                "bb_left": det.tlwh[0],
                "bb_top": det.tlwh[1], 
                "bb_width": det.tlwh[2],
                "bb_height": det.tlwh[3], 
                "conf": det.confidence, 
                "x": -1, "y": -1, "z": -1
			})
			bbox = track.to_tlbr() #Get the corrected/predicted bounding box
			id_num = str(track.track_id) #Get the ID for the particular track.
			features = track.features #Get the feature vector corresponding to the detection.

			#Draw bbox from tracker.
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
			cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

			#Draw bbox from detector. Just to compare.
			for det in detections_class:
				bbox = det.to_tlbr()
				cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
		
		cv2.imshow('frame',frame)
		out.write(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		frame_id+=1
	pd.DataFrame(data).sort_values(by=["frame"]).to_csv(csv_name, header=False, index=False)

