import sys  
import json 
from subprocess import Popen  
import mapdriver as md 
# import mapbox as md2
import esri as md2
import graph_ops as graphlib 
import math 
import numpy as np 
import scipy.misc 
from PIL import Image 
import pickle 

dataset_cfg = []
total_regions = 0 

tid = 0 #int(sys.argv[1])
tn = 1 #int(sys.argv[2])


for name_cfg in sys.argv[1:]:
	dataset_cfg_ = json.load(open(name_cfg, "r"))

	
	for item in dataset_cfg_:
		dataset_cfg.append(item)
		
		ilat = item["lat_n"]
		ilon = item["lon_n"]

		total_regions += ilat * ilon 


print("total regions", total_regions)

Popen("mkdir tmp", shell=True).wait()
#Popen("mkdir googlemap", shell=True).wait() 

# dataset_folder = "global_dataset_mapbox_no_service_road"
# folder_mapbox_cache = "cache_esri/"
dataset_folder = "chengdu"
folder_mapbox_cache = "cache_esri_chengdu/"

Popen("mkdir %s" % dataset_folder, shell=True).wait()
Popen("mkdir %s" % folder_mapbox_cache, shell=True).wait()

# download imagery and osm maps 

c = 0
tiles_needed = 0
size = 2048     # hhy change, size of an img and osm

for item in dataset_cfg:
	#prefix = item["cityname"]
	ilat = item["lat_n"]
	ilon = item["lon_n"]
	lat = item["lat"]
	lon = item["lon"]

	for i in range(ilat):
		for j in range(ilon):
			print(c, total_regions)

			if c % tn == tid:
				pass
			else:
				c = c + 1
				continue


			lat = item["lat"]
			lon = item["lon"]

			lat_st = lat + size/111111.0 * i
			lon_st = lon + size/111111.0 * j / math.cos(math.radians(lat))
			lat_ed = lat + size/111111.0 * (i+1)
			lon_ed = lon + size/111111.0 * (j+1) / math.cos(math.radians(lat))


			# download satellite imagery from google
			# if abs(lat_st) < 33:
			# 	zoom = 18
			# else:
			# 	zoom = 17

			# download satellite imagery from mapbox
			if abs(lat_st) < 33:
				zoom = 18
			else:
				zoom = 17

			print(lat_st, lon_st, lat_ed, lon_ed)
			

			# comment out the image downloading part 
			img, _ = md2.GetMapInRect(lat_st, lon_st, lat_ed, lon_ed, zoom=zoom, folder=folder_mapbox_cache)
			print(np.shape(img))

			# img = scipy.misc.imresize(img.astype(np.uint8), (size,size))    # scipy.mise被删除，改用PIL
			img = Image.fromarray(img.astype(np.uint8))
			img = img.resize((size, size), Image.BILINEAR)
			img = np.array(img)
			Image.fromarray(img).save(dataset_folder+"/region_%d_sat.png" % c)


			# download openstreetmap 
			OSMMap = md.OSMLoader([lat_st,lon_st,lat_ed,lon_ed], False, includeServiceRoad = False)

			node_neighbor = {} # continuous

			for node_id, node_info in OSMMap.nodedict.items():  # dict.iteritems()在python3舍弃
				lat = node_info["lat"]
				lon = node_info["lon"]

				n1key = (lat,lon)


				neighbors = []
				# for nid in node_info["to"].keys() + node_info["from"].keys() :    # python3的dict不可相加
				for nid in list(node_info["to"].keys()) + list(node_info["from"].keys()):
					if nid not in neighbors:
						neighbors.append(nid)

				for nid in neighbors:
					n2key = (OSMMap.nodedict[nid]["lat"],OSMMap.nodedict[nid]["lon"])

					node_neighbor = graphlib.graphInsert(node_neighbor, n1key, n2key)
					
			#graphlib.graphVis2048(node_neighbor,[lat_st,lon_st,lat_ed,lon_ed], "raw.png")
			
			# interpolate the graph (20 meters interval)
			node_neighbor = graphlib.graphDensify(node_neighbor)
			node_neighbor_region = graphlib.graph2RegionCoordinate(node_neighbor, [lat_st,lon_st,lat_ed,lon_ed], size)	# hhy modify size 2026-01-26
			prop_graph = "{}/region_{}_graph_gt.pickle".format(dataset_folder, c)
			pickle.dump(node_neighbor_region, open(prop_graph, "wb"))  # python3不能只以w打开，需要wb才能写二进制，否则只能写str

			#graphlib.graphVis2048(node_neighbor,[lat_st,lon_st,lat_ed,lon_ed], "dense.png")
			graphlib.graphVis2048Segmentation(node_neighbor, [lat_st,lon_st,lat_ed,lon_ed], f"{dataset_folder}/region_{c}_gt.png", size)

			node_neighbor_refine, sample_points = graphlib.graphGroundTruthPreProcess(node_neighbor_region)

			refine_graph = "{}/region_{}_refine_gt_graph.p".format(dataset_folder, c)
			pickle.dump(node_neighbor_refine, open(refine_graph, "wb"))     # 同理，python3改用wb
			json.dump(sample_points, open(f"{dataset_folder}/region_{c}_refine_gt_graph_samplepoints.json", "w"), indent=2)
			c+=1

