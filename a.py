
import os 

a = os.listdir('/mnt/s3fs/argo_data/val/182ba3f7-b89a-36cc-ae40-32a341b0d3e9/sensors/cameras/ring_rear_left')
b = '315966718557428274.jpg'
print(b in a)
