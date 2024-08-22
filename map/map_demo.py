import folium
from folium import plugins
import numpy as np
import pandas as pd
import requests
#
strategy = 5  # 规划策略 见https://lbs.amap.com/api/webservice/guide/api/direction#driving
gaode_key = 'f772729dcd5b1c2fbbc0e6942c6011ab"'  # 调用高德地图key
lat_and_lon = []  # 储存详细坐标经纬度
Lon = []  # 储存经度
Lat = []  # 储存纬度
df = pd.read_excel('d:/PYTHON/Manhole Cover Detection/map/总数据.xls', sheet_name='一区')
way_lon = df['经度'].tolist()
way_lat = df['纬度'].tolist()

waypoint = ''  # 路径点
jwd = df['经纬度'].tolist()  # 供应商经纬度

grouped_strings = []
tolls = 0
duration = 0


def get_routes(start, end, mode, gaode_key, waypoint):
    url = f'https://restapi.amap.com/v3/direction/driving?origin={start}&destination={end}&strategy={mode}&waypoints={waypoint}&key={gaode_key}'
    response = requests.get(url)
    data = response.json()

    if data['status'] == '1':
        routes = data['route']['paths'][0]['steps']
        return routes
    else:
        print('未获取到相关路径')
        return None


for i in range(0, len(jwd), 16):
    group = jwd[i:i + 16]
    string = ';'.join(map(str, group))
    start = group[0]
    end = group[-1]
    waypoint = string

    # 获取路线规划
    routes = get_routes(start, end, strategy, gaode_key, waypoint)

    if routes:
        for i, step in enumerate(routes):
            lat_and_lon.append(step["polyline"])
            tolls += int(step["tolls"])
            duration += int(step["duration"])
    else:
        print('无法获取路线规划。')

# 提取坐标点并全添加到 Lon 和 Lat 列表中
for item in lat_and_lon:
    points = item.split(';')
    for point in points:
        coords = point.split(',')
        Lon.append(float(coords[0]))
        Lat.append(float(coords[1]))


def Map( la, lo):
    tri = np.array(list(zip([23.243466, 113.637713], [23.943466, 113.637813])))
    # tri = np.[[23.243466, 113.637713],[23.243466, 113.637813]]
    print(tri)

    san_map = folium.Map(
        location=[1.243466, 60.637713],
        zoom_start=16,
        # 调用高德街道图
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
        # tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', # 高德卫星图
        attr='default')

    folium.PolyLine(tri, color='#ff0000').add_to(san_map)
    marker_cluster = plugins.MarkerCluster().add_to(san_map)
    for lat, lon in zip([23.243466, 113.637713], [23.943466, 113.637813]):
        icon = folium.Icon(color='red', icon='info-sign')  # 自定义图标
        folium.Marker([lat, lon], icon=icon).add_to(marker_cluster)
    san_map.save('地图.html')
    print("总花费{}元".format(tolls))
    print("耗费{}分钟".format(duration / 60))


def main():
    Map( way_lat, way_lon)
    # way_lat

if __name__ == '__main__':
    main()


