import cv2
from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO('yolov8n.pt')  # n=nano, s=small, m=medium, l=large, x=extra large

# 2. 对图片进行目标检测
results = model('/home/boran.lbr/program_code/gen_movie/api/woman_skyline_original_720p.jpeg')

# 3. 处理检测结果
for result in results:
    # 获取边界框
    boxes = result.boxes
    for box in boxes:
        # 获取坐标、置信度和类别
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = model.names[int(class_id)]
        
        print(f"检测到: {class_name}, 置信度: {confidence:.2f}")
    
    # 保存标注后的图片
    result.save('/home/boran.lbr/program_code/gen_movie/api/woman_skyline_original_720p_result.jpg')
