import yaml
import torch, os
# apiData = {
#     "page": 1,
#     "msg": "地址",
#     "data": [{
#         "id": 1,
#         "name": "学校"
#     }, {
#         "id": 2,
#         "name": "公寓"
#     }, {
#         "id": 3,
#         "name": "流动人口社区"
#     }],
# }
#
# # sort_keys=False字段表示不改变原数据的排序
# # allow_unicode=True 允许写入中文，必须以字节码格式写入
# with open("config.yaml", "w", encoding="utf-8") as fs:
#     yaml.safe_dump(data=apiData, stream=fs, sort_keys=False, allow_unicode=True)

weights = "../weights/yolov5s.pt"
ckpt = torch.load(weights)
state_dict = ckpt['model'].float().state_dict()
print(state_dict)
print(os.sep)