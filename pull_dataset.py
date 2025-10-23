from roboflow import Roboflow
rf = Roboflow()
project = rf.workspace().project("tree-detection-hgxhy-168bb")
dataset = project.version(1).download("yolov8")
