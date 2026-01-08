# 1. OSM

- [Open Street Map](https://www.openstreetmap.org/)
- [Geofabrik](https://download.geofabrik.de/)
- [China](https://download.geofabrik.de/asia/china.html)
- [raw directory index](https://download.geofabrik.de/asia/china.html#)

# 2. Target Region

## Xi'an

34.20, 108.91, 34.29, 109.01

34°12'0.0000"

108°54'36.0000"

34°17'24.0000"

109°0'36.0000"

Sijie Ruan code
- [tptk](https://github.com/sjruan/tptk)
- [osm2rn](https://github.com/sjruan/osm2rn)

```python
# osm2rn osm_clip.py
python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-140101.osm.pbf --output_path ./dataset/osm/xian-140101.osm.pbf --min_lat 34.20 --min_lng 108.91 --max_lat 34.29 --max_lng 109.01

python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-190101.osm.pbf --output_path ./dataset/osm/xian-190101.osm.pbf --min_lat 34.20 --min_lng 108.91 --max_lat 34.29 --max_lng 109.01

python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-140101.osm.pbf --output_path ./dataset/osm/xian-plus-140101.osm.pbf --min_lat 33.70 --min_lng 108.40 --max_lat 34.80 --max_lng 109.50

python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-190101.osm.pbf --output_path ./dataset/osm/xian-plus-190101.osm.pbf --min_lat 33.70 --min_lng 108.40 --max_lat 34.80 --max_lng 109.50

# osm2rn osm_to_rn.py

python ./utils/osm2rn/osm_to_rn.py --input_path ./dataset/osm/xian-140101.osm.pbf --output_path ./dataset/osm/xian140101

python ./utils/osm2rn/osm_to_rn.py --input_path ./dataset/osm/xian-190101.osm.pbf --output_path ./dataset/osm/xian190101

```

