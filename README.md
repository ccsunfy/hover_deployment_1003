# hover_deployment_1003
# 先运行segmotion_full生成桶的分割图：
1. python segmotion_full.py --ros --color blue (颜色一开始就定义好，这样策略只会识别对应颜色的桶，不会错降)
# 运行hover policy:
2. python searchland_run.py
策略预期会完成gps坐标范围附近桶的搜寻与悬停，最后输出标志位：/searchland/hover_success