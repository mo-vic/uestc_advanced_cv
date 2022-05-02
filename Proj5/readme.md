## 说明

**Note**：在`proj5/config`目录下的`yolov3-tiny.cfg`配置文件中修改超参数。



我修改了训练参数`batch=7`，并调整了以下NMS 参数：

```python
# original
detections = non_max_suppression(detections, 0.2, 0.7)
# mine
detections = non_max_suppression(detections, 0.25, 0.6)
```

