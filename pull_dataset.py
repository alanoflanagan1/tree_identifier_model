from roboflow import Roboflow

# Initialize Roboflow (replace with your key if needed)
# rf = Roboflow(api_key="XpDuyiXmQ3JFz0vBDcbz")  # optional if project is public
# project = rf.workspace().project("tree-detection-ekaot-jq24a")
# dataset = project.version(1).download("yolov8")  # downloads as YOLOv8 format

#tree-detection-ekaot-jq24a
#tree-detection-hgxhy-168bb
rf = Roboflow(api_key="XpDuyiXmQ3JFz0vBDcbz")
project = rf.workspace("phd-l0gzl").project("tree-detection-ekaot-jq24a")
version = project.version(1)
dataset = version.download("yolov8")