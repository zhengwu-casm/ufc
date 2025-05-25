# --------------------------------------------
# Data Preprocess
# Calculate indicators, select samples
# --------------------------------------------
from osgeo import gdal
from osgeo import ogr
import numpy as np

import geoutils
import sklearn.metrics, sklearn.decomposition
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPoint, Point
import json, math
from sklearn.linear_model import LinearRegression

from shapely.ops import voronoi_diagram

# gdal environment initialization
def init_gdal():
    # support for Chinese paths
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # support for Chinese Fields
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    # register all drivers
    ogr.RegisterAll()
    return

# Open the data source
def open_data_store(input_file, update):
    data_store = ogr.Open(input_file, update)
    if data_store is None:
        print("Open data {0} Failed！".format(input_file))
        return
    print("Open data {0} successed！".format(input_file))
    return data_store

# Open the layer
def open_layer(data_store, layer_index):
    layer = data_store.GetLayerByIndex(layer_index)
    if layer is None:
        print("Failed to get the layer！\n")
        return
    # Initialize the layer, clear layer status
    layer.ResetReading()
    return layer

# create polygon
def create_polygon(outring_coords_array, inring_coords_array=None):
    polygon = ogr.Geometry(ogr.wkbPolygon)
    outring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in outring_coords_array:
        outring.AddPoint(coord[0], coord[1])
    outring.CloseRings()
    polygon.AddGeometry(outring)
    if inring_coords_array is not None:
        inring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in inring_coords_array:
            inring.AddPoint(coord[0], coord[1])
        inring.CloseRings()
        polygon.AddGeometry(inring)
    return polygon

# Perform polygon conversion (from gdal to shapely)
def create_gdal_polygon_from_shapely_polygon(shapele_polygon):
    shapely_polygo_exterior = shapele_polygon.exterior
    coordinates = shapely_polygo_exterior.coords
    coords = []
    for i in range(coordinates.__len__()):
        (x, y) = coordinates.__getitem__(i)
        coords.append((x, y))
    gdal_polygon = create_polygon(coords)
    return gdal_polygon

# Get polygon coordinates
def get_coords_polygon(polygon):
    # print(polygon.GetGeometryCount())
    outring = polygon.GetGeometryRef(0)
    # inring = polygon.GetGeometryRef(1)
    coords = []
    if outring is not None:
        # print(outring)
        for i in range(outring.GetPointCount()):
            # point = outring.GetPoint(i)
            coord = [outring.GetX(i), outring.GetY(i)]
            coords.append(coord)

    # if inring is not None:
    #     print(inring)
    return np.array(coords)

# Get polygon coordinates (tuple format)
def get_coords_seq_polygon(polygon):
    # print(polygon.GetGeometryCount())
    outring = polygon.GetGeometryRef(0)
    # inring = polygon.GetGeometryRef(1)
    coords = []
    if outring is not None:
        # print(outring)
        for i in range(outring.GetPointCount()):
            # point = outring.GetPoint(i)
            coord = (outring.GetX(i), outring.GetY(i))
            coords.append(coord)
    coords = tuple(coords)
    # if inring is not None:
    #     print(inring)
    return np.array(coords)

# Calculation of building indicators
def calculate_indicator(building_coords):
    # Calculate the basic indicators of polygon.

    # --------------------------------------------
    # 1.Size indicators
    # Area, Perimeter, long_chord, mean_radius
    # --------------------------------------------

    # Geometry descriptors: Size-Area:area, Size-Perimeter:peri smbr_area(mbr area), Position:Centriod:CX/CY
    [[CX, CY], area, peri] = geoutils.get_basic_parametries_of_Poly(building_coords)

    # obb(minimum bounding box), smbr_area(minimum bounding box area)
    obb, smbr_area = geoutils.mininumAreaRectangle(building_coords)
    # orientation：SBRO(Minimum bounding box orientation)
    orientation = obb.Orientation()
    # Shape：Elongation
    length_width = obb.e1 / obb.e0 if obb.e0 > obb.e1 else obb.e0 / obb.e1

    # Regularization operations on point coordinates
    uniform_coords = np.array([[(j[0] - CX), (j[1] - CY)] for j in building_coords])
    uniform_size = len(uniform_coords)
    # Closing the polygon
    if uniform_coords[0][0] - uniform_coords[uniform_size - 1][0] != 0 or uniform_coords[0][1] - \
            uniform_coords[uniform_size - 1][1] != 0:
        print('Closing!')
        uniform_coords.append(uniform_coords[0])

    convexhull = ConvexHull(uniform_coords)

    sum_radius, size_radius, mean_radius, long_chord = 0, 0, 0, 0
    # compute Mean radius
    for j in range(0, uniform_size - 1):
        sum_radius += math.sqrt(
            uniform_coords[j][0] * uniform_coords[j][0] + uniform_coords[j][1] * uniform_coords[j][1])
        size_radius += 1
    if size_radius != 0:
        mean_radius = sum_radius / size_radius

    # compute long_chord
    # Calculate the distance between any two points of a polygon
    pairwise_distances, index_j, index_h = sklearn.metrics.pairwise.pairwise_distances(
        uniform_coords[convexhull.vertices], metric="euclidean", n_jobs=1), 0, 0
    # Find the longest distance between two points as the longest chord
    for j in range(0, len(pairwise_distances)):
        for h in range(j, len(pairwise_distances)):
            if (pairwise_distances[j, h] > long_chord):
                long_chord, index_j, index_h = pairwise_distances[j, h], j, h

    # Find the distance between the two points of the second longest as the second longest chord
    second_chord, index_p, index_q = 0, 0, 0
    for j in range(0, len(pairwise_distances)):
        for h in range(j, len(pairwise_distances)):
            if pairwise_distances[j, h] > second_chord:
                if j != index_j and h != index_h:
                    second_chord, index_p, index_q = pairwise_distances[j, h], j, h

    # --------------------------------------------
    # 2.Orientation indicators
    # Smallest Bounding Rectangle Orientation(SBRO), Long Chord Orientation (LCO), Bisector Orientation (BISO)
    # WSWO (WSWO)
    # --------------------------------------------

    # Calculate the longest chord orientation(LCO)
    from_longedge, to_longedge = uniform_coords[convexhull.vertices[index_j]], uniform_coords[
        convexhull.vertices[index_h]]
    longedge_orien = abs(math.atan2(from_longedge[0] - to_longedge[0], from_longedge[1] - to_longedge[1]))
    # Calculate the direction of the second longest chord
    from_secondedge, to_secondedge = uniform_coords[convexhull.vertices[index_p]], uniform_coords[
        convexhull.vertices[index_q]]
    secondedge_orien = abs(
        math.atan2(from_secondedge[0] - to_secondedge[0], from_secondedge[1] - to_secondedge[1]))

    # Calculate BISO
    bissector_orien = (longedge_orien * long_chord + secondedge_orien * second_chord) / (
            long_chord + second_chord)

    # Calculate SBRO/WSWO
    smbr_orien, wall_orien, weight_orien = orientation, 0, 0
    longedge_a, longedge_b, longedge_c = geoutils.get_equation(from_longedge, to_longedge)
    long_width, up_offset, down_offset = 0, longedge_c, longedge_c
    for j in range(0, uniform_size - 1):
        crossing_product = longedge_a * uniform_coords[j][0] + longedge_b * uniform_coords[j][1]
        if crossing_product + up_offset < 0:
            up_offset = -crossing_product
        if crossing_product + down_offset > 0:
            down_offset = -crossing_product
    longedge_square = math.sqrt(longedge_a * longedge_a + longedge_b * longedge_b)
    if longedge_square == 0:
        long_width = abs(up_offset - down_offset)
    else:
        long_width = abs(up_offset - down_offset) / longedge_square

    edge_orien_weight, edge_length_sun, edge_tuple, candidate_max = 0, 0, [], 0
    for j in range(0, uniform_size - 1):
        dx, dy = uniform_coords[j + 1][0] - uniform_coords[j][0], uniform_coords[j + 1][1] - \
                 uniform_coords[j][1]
        edge_orien = (math.atan2(dx, dy) + math.pi) % (math.pi / 2.0)
        edge_length = math.sqrt(dx * dx + dy * dy)

        edge_orien_weight += edge_length * edge_orien
        edge_length_sun += edge_length

        edge_tuple.append([edge_orien, edge_length])

    for j in range(0, 90):
        candidate_orien, candidate_weight = j * math.pi / 180, 0
        for m in range(0, len(edge_tuple)):
            if abs(edge_tuple[m][0] - candidate_orien) < math.pi / 24:
                candidate_weight += (math.pi / 24 - abs(edge_tuple[m][0] - candidate_orien)) * \
                                    edge_tuple[m][1] / (math.pi / 24)
        if candidate_weight > candidate_max:
            candidate_max, weight_orien = candidate_weight, candidate_orien

    # --------------------------------------------
    # 3. Shape indicators
    # RIC Compactness (RICC), IPQ Compactness (IPQC), FRA Compactness (FRAC), GIB Compactness (GIBC),
    # DIV Compactness (DIVC), Elongation, Ellipticity, Concavity, DCM, BOT, BOY, Eccentricity, Central Moment 11 (CM11)
    # --------------------------------------------
    # Calculate RICC, IPQC, FRAC
    RIC_compa, IPQ_compa, FRA_compa = peri / area, 4 * math.pi * area / (peri * peri), 1 - math.log(
        area) * .5 / math.log(peri)
    # Calculate GIBC, DIVC
    GIB_compa, Div_compa = 2 * math.sqrt(math.pi * area) / long_chord, 4 * area / (long_chord * peri)

    # Calculate elongation, ellipticity, concavity
    elongation, ellipticity, concavity = length_width, long_width / long_chord, area / convexhull.area

    # Calculate DCM, BOT
    radius, standard_circle, enclosing_circle = math.sqrt(area / math.pi), [], geoutils.make_circle(
        uniform_coords)
    for j in range(0, 60):
        standard_circle.append(
            [math.cos(2 * math.pi * j / 60) * radius, math.sin(2 * math.pi * j / 60) * radius])

    standard_intersection = Polygon(uniform_coords).intersection(Polygon(standard_circle))

    DCM_index = area / (math.pi * enclosing_circle[2] * enclosing_circle[2])
    BOT_index = 1 - standard_intersection.area / area

    # Calculate BOY
    closest_length, closest_sum, closest_size, BOY_measure = [], 0, 0, 0
    for j in range(0, 40):
        x, y = math.cos(2 * math.pi * j / 40) * peri, math.sin(2 * math.pi * j / 40) * peri
        closest_point = geoutils.find_intersection(uniform_coords, [x, y])

        if closest_point is not None:
            closest_length.append(
                math.sqrt(closest_point[0] * closest_point[0] + closest_point[1] * closest_point[1]))
            closest_sum += math.sqrt(
                closest_point[0] * closest_point[0] + closest_point[1] * closest_point[1])
            closest_size += 1

    for j in closest_length:
        BOY_measure += abs(100 * j / closest_sum - 100 / closest_size)
    BOY_index = 1 - BOY_measure / 200

    # Calculate M11
    M02, M20, M11 = 0, 0, 0
    for j in range(0, uniform_size - 1):
        M02 += (uniform_coords[j][1]) * (uniform_coords[j][1])
        M20 += (uniform_coords[j][0]) * (uniform_coords[j][0])
        M11 += (uniform_coords[j][0]) * (uniform_coords[j][1])

    # Calculate Eccentricity
    Eccentricity = ((M02 - M20) * (M02 - M20) + 4 * M11) / area

    # compositing the descriptors of Size, Orientation and Shape as Geometry feature characteristics.
    geo_features = [area, peri, long_chord, mean_radius, \
                    smbr_orien, longedge_orien, bissector_orien, weight_orien, \
                    RIC_compa, IPQ_compa, FRA_compa, GIB_compa, Div_compa, \
                    elongation, ellipticity, concavity, DCM_index, BOT_index, BOY_index, \
                    M11, Eccentricity]

    return geo_features

# Fitting a linear trendline
def linear_trend(x, y):
    lreg = LinearRegression()
    x = x.reshape(len(x),1)
    lreg.fit(x, y)
    coef = round(lreg.coef_[0], 4)
    intercept = round(lreg.intercept_, 4)

    # pred = lreg.predict(x)
    # plt.plot(pred)
    # plt.title("trend yt={} + ({})t".format(intercept, coef))
    # plt.show()
    return intercept, coef

#  Calculate the population heat characteristics of urban functional zone
def calculate_pop_indicator(yard_feature, building_layer, pop_layer):
    day_pop_density, night_pop_density, day_pop_density_rate, night_pop_density_rate, pop_density_workday_weekend_diff = 0., 0., 0., 0., 0.
    yard_geometry = yard_feature.GetGeometryRef()
    yard_centroid = yard_geometry.Centroid()
    # building_layer.ResetReading()
    # building_layer.SetSpatialFilter(yard_geometry)
    # building_feature = building_layer.GetNextFeature()
    # building_area_statistic = 0.
    # while building_feature is not None:
    #     building_area_statistic += building_feature.GetGeometryRef().GetArea()
    #     building_feature = building_layer.GetNextFeature()

    pop_layer.ResetReading()
    pop_layer.SetSpatialFilter(yard_centroid.Buffer(1.))
    # 有可能找不到对应的gird
    pop_feature = pop_layer.GetNextFeature()
    # flag = False
    if pop_feature is None:
        # flag = True
        pop_layer.ResetReading()
        pop_layer.SetSpatialFilter(yard_geometry)
        pop_feature = pop_layer.GetNextFeature()
    if pop_feature is None:
        return None
    pop_feature_defn = pop_feature.GetDefnRef()
    h08_field_index = pop_feature_defn.GetFieldIndex("h08")
    h09_field_index = pop_feature_defn.GetFieldIndex("h09")
    h10_field_index = pop_feature_defn.GetFieldIndex("h10")
    h11_field_index = pop_feature_defn.GetFieldIndex("h11")
    h12_field_index = pop_feature_defn.GetFieldIndex("h12")


    h16_field_index = pop_feature_defn.GetFieldIndex("h16")
    h17_field_index = pop_feature_defn.GetFieldIndex("h17")
    h18_field_index = pop_feature_defn.GetFieldIndex("h18")
    h19_field_index = pop_feature_defn.GetFieldIndex("h19")
    h20_field_index = pop_feature_defn.GetFieldIndex("h20")
    area_max = 0.
    while pop_feature is not None:
        grid_geometry = pop_feature.GetGeometryRef()
        intersect_geometry = grid_geometry.Intersection(yard_geometry)
        intersect_area = intersect_geometry.GetArea()
        if intersect_area < area_max:
            pop_feature.Destroy()
            pop_feature = pop_layer.GetNextFeature()
            continue
        area_max = intersect_area
        h08_field_value = pop_feature.GetFieldAsDouble(h08_field_index)
        h09_field_value = pop_feature.GetFieldAsDouble(h09_field_index)
        h10_field_value = pop_feature.GetFieldAsDouble(h10_field_index)
        h11_field_value = pop_feature.GetFieldAsDouble(h11_field_index)
        h12_field_value = pop_feature.GetFieldAsDouble(h12_field_index)

        h16_field_value = pop_feature.GetFieldAsDouble(h16_field_index)
        h17_field_value = pop_feature.GetFieldAsDouble(h17_field_index)
        h18_field_value = pop_feature.GetFieldAsDouble(h18_field_index)
        h19_field_value = pop_feature.GetFieldAsDouble(h19_field_index)
        h20_field_value = pop_feature.GetFieldAsDouble(h20_field_index)
        # 早上10点、晚上8点人口密度
        day_pop_density, night_pop_density = h10_field_value / grid_geometry.GetArea(), h20_field_value / grid_geometry.GetArea()
        # h10_density = h10_field_value / building_area_statistic
        # h20_density = h20_field_value / building_area_statistic
        # 上午/下午人口变化趋势
        day_y = np.array([h08_field_value, h09_field_value, h10_field_value, h11_field_value, h12_field_value])
        day_y /= grid_geometry.GetArea()
        # day_y -= np.mean(day_y)
        day_x = np.array([8, 9, 10, 11, 12])
        day_x -= np.min(day_x)-1

        night_y = np.array([h16_field_value, h17_field_value, h18_field_value, h19_field_value, h20_field_value])
        night_y /= grid_geometry.GetArea()
        # night_y -= np.mean(night_y)
        night_x = np.array([16, 17, 18, 19, 20])
        night_x -= np.min(night_x)-1

        # 计算回归系数
        # day_slope, day_intercept = np.polyfit(day_x, day_y, 1)
        # print(day_slope, day_intercept)
        day_intercept, day_slope = linear_trend(day_x, day_y)
        # print(day_slope, day_intercept)
        # night_slope, night_intercept = np.polyfit(night_x, night_y, 1)
        night_intercept, night_slope = linear_trend(night_x, night_y)
        # print(night_slope, night_intercept)
        day_pop_density_rate, night_pop_density_rate = day_slope, night_slope

        pop_feature.Destroy()
        pop_feature = pop_layer.GetNextFeature()
    return day_pop_density, night_pop_density, day_pop_density_rate, night_pop_density_rate

# Functional area categories
def function_classification():
    # "reside": reside;"life": shopping, consumption, entertainment, recreation, sports; "work":study, work;
    function = ["reside", "life", "work", "unknow"]
    return function

# Functional area category index
def function_Index(function):
    function_idx = {"reside": 0, "life": 1, "work": 2, "unknow": -1}
    return function_idx[function]

# Functional area day/night type
def day_night_classification():
    day_night = {"reside": "daynight", "life": "daynight", "work": "day"}
    return day_night

# POI type mapping to function area type
def poi_classification():
    # mapping relationship
    poi_class = {"餐饮服务": "life", "道路附属设施": "life", "地名地址信息;交通地名": "life", \
                 "交通设施服务": "life", "地名地址信息;地名地址信息": "reside", "商务住宅": "reside", \
                 "地名地址信息;普通地名": "reside", "地名地址信息;门牌信息": "reside", "住宿服务": "reside", \
                 "住宿服务;住宿服务相关": "reside", "住宿服务;旅馆招待所": "reside", "住宿服务;宾馆酒店": "reside", \
                 "地名地址信息;自然地名": "life", "风景名胜": "life", "公司企业": "work", "金融保险服务": "work", \
                 "购物服务": "life", "生活服务": "life", "摩托车服务": "life", "汽车服务": "life", "汽车维修": "life", \
                 "汽车销售": "life", "公共设施": "life", "科教文化服务": "work", "体育休闲服务": "life", \
                 "医疗保健服务": "work", "政府机构及社会团体": "work"}
    return poi_class

# POI type reclassification
def reclassify(type):
    new_type = None
    type_list = type.split(";")
    if type_list[0] == "地名地址信息":
        new_type = type_list[0] + ";" + type_list[1]
    elif type_list[0] == "住宿服务" and len(type_list) > 1:
        new_type = type_list[0] + ";" + type_list[1]
    else:
        new_type = type_list[0]
    poi_class = poi_classification()
    funtion = poi_class.get(new_type)
    return funtion

# Calculate POI Heat
def calculate_poi_indicator(yard_feature, pop_layer, poi_layer):
    yard_geometry = yard_feature.GetGeometryRef()
    # yard_envelope = yard_geometry.GetEnvelope()
    # print(yard_envelope)
    # S_i
    yard_area = yard_geometry.GetArea()

    # Percentage of quantity in each category of POI, poi area, Heat of poi weighted by heat of circadian population
    # X_ij = (A_ij∙C_ij∙S_ij) / S_i
    # Population activity during the day and night: A_ij, Number of POI points: C_ij,
    # Tyson polygon area for function class j in courtyard unit i: S_ij, Area of courtyard unit i: S_i为街区单元i的面积

    # Counting the number of POIs per category
    poi_layer.ResetReading()
    poi_layer.SetSpatialFilter(yard_geometry)
    poi_feature = poi_layer.GetNextFeature()
    if poi_feature is None:
        return None
    poi_feature_defn = poi_feature.GetDefnRef()
    type_field_index = poi_feature_defn.GetFieldIndex("type")
    name_field_index = poi_feature_defn.GetFieldIndex("name")
    poi_area_stat, poi_count_stat, poi_pop_stat, poi_coords, poi_functions = {}, {}, {}, [], []
    pop_day, pop_night, pop_day_night = 0, 0, 0
    day_night = day_night_classification()
    poi_pairs = {}
    # for i in range(len(poi_points)):
    #     if test_point_pairs.get(poi_points[i][0]) is None:
    #         test_point_pairs[poi_points[i][0]] = poi_points[i][1]
    #     else:
    #         if test_point_pairs[poi_points[i][0]] == poi_points[i][1]:
    #             print("find error!")
    while poi_feature is not None:
        type = poi_feature.GetFieldAsString(type_field_index)
        # name = poi_feature.GetFieldAsString(name_field_index)
        # print(type)
        function = reclassify(type)
        if function is None:
            poi_feature.Destroy()
            poi_feature = poi_layer.GetNextFeature()
            continue
        poi_functions.append(function)
        # Calculate C_ij
        if poi_count_stat.get(function) is None:
            poi_count_stat[function] = 1
        else:
            poi_count_stat[function] += 1
        poi_point = poi_feature.GetGeometryRef()
        if poi_point.GetGeometryType() == ogr.wkbMultiPoint or poi_point.GetGeometryType() == ogr.wkbMultiPointM or \
                poi_point.GetGeometryType() == ogr.wkbMultiPointZM or \
                poi_point.GetGeometryType() == ogr.wkbMultiPoint25D:
            poi_point = poi_point.GetGeometryRef(0)
        coords = [poi_point.GetX(0), poi_point.GetY(0)]

        # Determine if a point is duplicated in position
        if poi_pairs.get(coords[0]) is None:
            poi_pairs[coords[0]] = coords[1]
        else:
            if poi_pairs[coords[0]] == coords[1]:
                # print(name)
                poi_feature.Destroy()
                poi_feature = poi_layer.GetNextFeature()
                continue
        # if poi_pairs.get(coords[0]) is None:
        #     poi_pairs[coords[0]] = coords[1]
        # else:
        #     if poi_pairs[coords[0]] == coords[1]:
        #         print(name)

        poi_coords.append(coords)

        # Calculation of day and night population
        pop_layer.ResetReading()
        pop_layer.SetSpatialFilter(poi_point.Buffer(0.5))
        pop_feature = pop_layer.GetNextFeature()
        if pop_feature is None:
            poi_feature.Destroy()
            poi_feature = poi_layer.GetNextFeature()
            continue
        pop_feature_defn = pop_feature.GetDefnRef()
        day_field_index = pop_feature_defn.GetFieldIndex("h10")
        night_field_index = pop_feature_defn.GetFieldIndex("h02")
        # print(function)
        if day_night[function] == "day":
            if poi_pop_stat.get('day') is None:
                poi_pop_stat['day'] = 1
            else:
                poi_pop_stat['day'] += 1
            # day
            pop_day += pop_feature.GetFieldAsInteger(day_field_index)
        elif day_night[function] == "night":
            if poi_pop_stat.get('night') is None:
                poi_pop_stat['night'] = 1
            else:
                poi_pop_stat['night'] += 1
            # night
            pop_night += pop_feature.GetFieldAsInteger(night_field_index)
        else:
            if poi_pop_stat.get('daynight') is None:
                poi_pop_stat['daynight'] = 1
            else:
                poi_pop_stat['daynight'] += 1
            pop_day_night += pop_feature.GetFieldAsInteger(day_field_index)
            pop_day_night += pop_feature.GetFieldAsInteger(night_field_index)
        # pop_feature_defn.Destroy()
        pop_feature.Destroy()
        poi_feature.Destroy()
        poi_feature = poi_layer.GetNextFeature()

    # A_ij
    if poi_pop_stat.get('day') is None:
        poi_pop_stat['day'] = 0.
    else:
        poi_pop_stat['day'] = pop_day / (250*250) / poi_pop_stat['day']
    if poi_pop_stat.get('night') is None:
        poi_pop_stat['night'] = 0.
    else:
        poi_pop_stat['night'] = pop_night / (250*250) / poi_pop_stat['night']
    if poi_pop_stat.get('daynight') is None:
        poi_pop_stat['daynight'] = 0.
    else:
        poi_pop_stat['daynight'] = pop_day_night / (250*250) / poi_pop_stat['daynight']

    poi_points = np.array(poi_coords)

    # Calculate S_ij
    if len(poi_points) > 3:
        # print(poi_points)
        # Determine if there are duplicate points
        # test_point_pairs = {}
        # for i in range(len(poi_points)):
        #     if test_point_pairs.get(poi_points[i][0]) is None:
        #         test_point_pairs[poi_points[i][0]] = poi_points[i][1]
        #     else:
        #         if test_point_pairs[poi_points[i][0]] == poi_points[i][1]:
        #             print("find error!")

        multipoint = MultiPoint(poi_points)
        area_shape_coords = get_coords_seq_polygon(yard_geometry)
        area_shape = Polygon(area_shape_coords)

        regions = voronoi_diagram(multipoint, area_shape)
        # print(regions)
        # print(multipoint)
        # fig = plt.figure(1, figsize=SIZE, dpi=90)
        # fig = plt.figure(1, dpi=90)
        #
        # ax = fig.add_subplot(111)
        #
        # for region in regions.geoms:
        #     # plot_polygon(region, ax=ax, add_points=False, color=BLUE)
        #     plot_polygon(region, ax=ax, add_points=False)
        #
        # # plot_points(multipoint, ax=ax, color=GRAY)
        # plot_points(multipoint, ax=ax)
        #
        # # plt.set_limits(ax, -1, 4, -1, 3)
        #
        # plt.show()

        for region in regions.geoms:
            index = None
            point_set = set()
            for i in range(len(poi_points)):
                point = Point(poi_points[i][0], poi_points[i][1])
                if point_set.__contains__((point.x, point.y)):
                    print("building center repeat point:{}".format(point.wkt))
                    continue
                else:
                    point_set.add((point.x, point.y))
                if region.contains(point):
                    index = i
                    break

            voronoi_geometry = create_gdal_polygon_from_shapely_polygon(region)
            voronoi_geometry_intersect = voronoi_geometry.Intersection(yard_geometry)
            voronoi_area = voronoi_geometry_intersect.GetArea()
            function = poi_functions[index]
            if poi_area_stat.get(function) is None:
                poi_area_stat[function] = voronoi_area
            else:
                poi_area_stat[function] += voronoi_area
            voronoi_geometry.Destroy()
            voronoi_geometry_intersect.Destroy()
    elif len(poi_points) > 0:
        for index in range(len(poi_points)):
            voronoi_area = yard_area / len(poi_points)
            function = poi_functions[index]
            if poi_area_stat.get(function) is None:
                poi_area_stat[function] = voronoi_area
            else:
                poi_area_stat[function] += voronoi_area
    else:
        return None

    # Calculate X_ij
    X_ij = {}
    function_dic = day_night.keys()
    for function in function_dic:
        if poi_area_stat.get(function) is None:
            X_ij[function] = 0.
            continue
        a_ij = 0.
        if day_night[function] == "day":
            a_ij = poi_pop_stat['day']
        elif day_night[function] == "night":
            a_ij = poi_pop_stat['night']
        else:
            a_ij = poi_pop_stat['daynight']

        X_ij[function] = a_ij * poi_count_stat[function] * poi_area_stat[function] / yard_area

    return X_ij

# Calculate the indicators, output json format
def calculate_indicators_json(building_file, use_building_area_limit, building_area_limit, yard_file, use_yard_area_limit, reside_yard_area_limit, work_yard_area_limit, pop_workday_file, pop_weekend_file, poi_file, output_json_file):
    # Initializing the GDAL Configuration
    init_gdal()
    # Read shp data
    # Open data
    # Open the building layer
    building_data_store = open_data_store(building_file, 0)
    if building_data_store is None:
        print("Failed to open building data！\n")
        return
    # Get building layer
    building_layer = open_layer(building_data_store, 0)
    if building_layer is None:
        print("Failed to get building layer！\n")
        return
    # Getting the courtyard layer
    yard_data_store = open_data_store(yard_file, 0)
    if yard_data_store is None:
        print("Failed to open courtyard data！\n")
        return
    yard_layer = open_layer(yard_data_store, 0)
    if yard_layer is None:
        print("Failed to get the courtyard layer！\n")
        return

    # Get Population Layer
    pop_workday_data_store = open_data_store(pop_workday_file, 0)
    if pop_workday_data_store is None:
        print("Failed to open weekday population data！\n")
        return
    pop_workday_layer = open_layer(pop_workday_data_store, 0)
    if pop_workday_layer is None:
        print("Failed to get workday population layer！\n")
        return
    pop_weekend_data_store = open_data_store(pop_weekend_file, 0)
    if pop_weekend_data_store is None:
        print("Failed to open rest day population data！\n")
        return
    pop_weekend_layer = open_layer(pop_weekend_data_store, 0)
    if pop_weekend_layer is None:
        print("Failed to get rest day population layer！\n")
        return

    # Get POI layer
    poi_data_store = open_data_store(poi_file, 0)
    if poi_data_store is None:
        print("Failed to open POI data！\n")
        return
    poi_layer = open_layer(poi_data_store, 0)
    if poi_layer is None:
        print("Failed to get POI layer！\n")
        return

    process_count, interpretedDic = 0, {}
    # Grouped by courtyard, each group constitutes a graph
    if use_yard_area_limit:
        yard_attribute_filter = ("(\"area\" > " + str(reside_yard_area_limit) +
                                 " and \"land_use\" = 'reside') or (\"area\" > " + str(work_yard_area_limit) +
                                 " and \"land_use\" = 'work') or ( \"land_use\" = 'life')")
        yard_layer.SetAttributeFilter(yard_attribute_filter)

    yard_feature = yard_layer.GetNextFeature()
    feature_defn = yard_feature.GetDefnRef()
    label_field_index = feature_defn.GetFieldIndex("label")
    # Start traversing the elements in the layer
    counter = 0
    while yard_feature is not None:
        counter += 1
        print("处理第{}个院落".format(counter))
        label = yard_feature.GetFieldAsInteger(label_field_index)
        #####################################################
        # Calculation of the population factor
        # Population density at 10:00 a.m. and 8:00 p.m. on weekdays,
        # and rate of change in morning and evening population density

        day_workday_pop_indicator = calculate_pop_indicator(yard_feature, \
                                                                     building_layer, \
                                                                     pop_workday_layer)
        if day_workday_pop_indicator is None:
            yard_feature.Destroy()
            yard_feature = yard_layer.GetNextFeature()
            continue
        [day_workday_pop_density, night_workday_pop_density, day_workday_pop_density_rate, \
         night_workday_pop_density_rate] = day_workday_pop_indicator
        # Population density at 10 a.m. and 8 p.m. on rest days,
        # and rate of change in morning and evening population density
        day_weekend_pop_indicator = calculate_pop_indicator(yard_feature, \
                                                                     building_layer, \
                                                                     pop_weekend_layer)
        if day_weekend_pop_indicator is None:
            yard_feature.Destroy()
            yard_feature = yard_layer.GetNextFeature()
            continue

        [day_weekend_pop_density, night_weekend_pop_density, day_weekend_pop_density_rate, \
         night_weekend_pop_density_rate] = day_weekend_pop_indicator

        # Difference in change in population density on weekdays/ weekends
        pop_density_workday_weekend_diff = day_workday_pop_density - day_weekend_pop_density

        #####################################################
        #####################################################
        # Calculating the POI factor
        poi_heater = calculate_poi_indicator(yard_feature, pop_workday_layer, poi_layer)
        if poi_heater is None:
            # for building_feature_item in building_group_list:
            #     building_feature_item.Destroy()
            yard_feature.Destroy()
            yard_feature = yard_layer.GetNextFeature()
            continue
        # [reside, life, company, traffic, education, healthcare, government, entertain, tour] = \
        #     [poi_heater['reside'], poi_heater['life'], poi_heater['company'], poi_heater['traffic'],\
        #      poi_heater['education'], poi_heater['healthcare'], poi_heater['government'], \
        #      poi_heater['entertain'], poi_heater['tour']]
        # [reside, publiclife, company] = \
        #     [poi_heater['reside'], poi_heater['publiclife'], poi_heater['company']]
        #
        # poi_heater_sort = [reside, publiclife, company]

        # [reside, life, company, public, tour] = \
        #     [poi_heater['reside'], poi_heater['life'], poi_heater['company'], poi_heater['public'], poi_heater['tour']]
        #
        # poi_heater_sort = [reside, life, company, public, tour]

        [reside, life, work] = \
            [poi_heater['reside'], poi_heater['life'], poi_heater['work']]

        poi_heater_sort = [reside, life, work]

        max_heater_index = np.argmax(poi_heater_sort)
        if max_heater_index != label:
            # print(max_heater_index, label, poi_heater_sort)
            print("Processing of {} courtyard:".format(counter), max_heater_index, label, poi_heater_sort)
            # yard_feature.Destroy()
            # yard_feature = yard_layer.GetNextFeature()
            # continue
        #####################################################

        gourp_id = int(yard_feature.GetFID())
        # label = yard_feature.GetFieldAsInteger(label_field_index)
        yard_geometry = yard_feature.GetGeometryRef()
        yard_geometry_buffer = yard_geometry.Buffer(-1.0)
        building_layer.SetSpatialFilter(yard_geometry_buffer)
        # Filtering of buildings
        if use_building_area_limit:
            attribute_filter = "\"Area\" > " + str(building_area_limit)
            building_layer.SetAttributeFilter(attribute_filter)

        building_group_list, total_area, total_peri, avg_area, avg_peri = [], 0., 0., 0., 0.
        building_feature = building_layer.GetNextFeature()
        while building_feature is not None:
            # Calculate area
            building_geometry = building_feature.GetGeometryRef()
            if not yard_geometry_buffer.Intersect(building_geometry):
                print("find not really intersects")
                continue

            building_area = building_geometry.GetArea()
            total_area += building_area
            building_coords = get_coords_polygon(building_geometry)
            coord_x, coord_y, building_peri = 0., 0., 0.
            for i in range(0, len(building_coords) - 1):
                coord_x += building_coords[i][0]
                coord_y += building_coords[i][1]
                building_peri += math.sqrt(pow(building_coords[i + 1][0] - building_coords[i][0], 2) +
                                  pow(building_coords[i + 1][1] - building_coords[i][1], 2))
            total_peri += building_peri

            building_group_list.append(building_feature)
            building_feature = building_layer.GetNextFeature()
        if len(building_group_list) > 0.:
            avg_area = total_area / len(building_group_list)
            avg_peri = total_peri / len(building_group_list)
        # Get points and edges
        centroid_points = []
        for building_feature_item in building_group_list:
            building_geometry = building_feature_item.GetGeometryRef()
            building_geometry_buffer = building_geometry.Buffer(-0.5)
            building_centroid = building_geometry_buffer.Centroid()
            coords = [building_centroid.GetX(0), building_centroid.GetY(0)]
            # print(coords)
            centroid_points.append(coords)
        centroid_points = np.array(centroid_points)
        vertex_coords, vertex_features = [], []
        if len(centroid_points) > 3:
            # generate voronoi diagram
            multipoint = MultiPoint(centroid_points)
            area_shape_coords = get_coords_seq_polygon(yard_geometry)
            area_shape = Polygon(area_shape_coords)

            regions = voronoi_diagram(multipoint, area_shape)
            # print(regions)
            # print(multipoint)
            # fig = plt.figure(1, figsize=SIZE, dpi=90)
            # fig = plt.figure(1, dpi=90)
            #
            # ax = fig.add_subplot(111)
            #
            # for region in regions.geoms:
            #     # plot_polygon(region, ax=ax, add_points=False, color=BLUE)
            #     plot_polygon(region, ax=ax, add_points=False)
            #
            # # plot_points(multipoint, ax=ax, color=GRAY)
            # plot_points(multipoint, ax=ax)
            #
            # # plt.set_limits(ax, -1, 4, -1, 3)
            #
            # plt.show()

            # vertex_coords, vertex_features = [], []
            for region in regions.geoms:
                index = None
                point_set = set()
                for i in range(len(centroid_points)):
                    point = Point(centroid_points[i][0], centroid_points[i][1])
                    if point_set.__contains__((point.x, point.y)):
                        print("building center repeat point:{}".format(point.wkt))
                        continue
                    else:
                        point_set.add((point.x, point.y))
                    if region.contains(point):
                        index = i
                        break

                voronoi_geometry = create_gdal_polygon_from_shapely_polygon(region)
                voronoi_geometry_intersect = voronoi_geometry.Intersection(yard_geometry)

                # Calculate building area
                building = building_group_list[index]
                building_geometry = building.GetGeometryRef()
                building_geometry_buffer = building_geometry.Buffer(-0.5)
                building_area = building_geometry_buffer.GetArea()
                building_centroid_point = building_geometry_buffer.Centroid()
                cx, cy = building_centroid_point.GetX(0), building_centroid_point.GetY(0)
                # Calculate Impact Area Ratio (IMA)
                voronoi_area = voronoi_geometry_intersect.GetArea()
                density = 0.
                if voronoi_area > 0.01:
                    density = building_area / voronoi_area
                else:
                    print("Processing the voronoi area of the {}th courtyard:".format(counter), voronoi_area)

                # Calculate Count Area Ratio
                count_density = len(building_group_list) / yard_geometry.GetArea()

                subobject = get_coords_polygon(building_geometry)
                ######################################################################

                geo_features = calculate_indicator(subobject)

                geo_features.append(density)

                geo_features.append(count_density)

                geo_features.append(day_workday_pop_density)
                geo_features.append(night_workday_pop_density)
                geo_features.append(day_workday_pop_density_rate)
                geo_features.append(night_workday_pop_density_rate)
                geo_features.append(day_weekend_pop_density_rate)
                geo_features.append(night_weekend_pop_density_rate)
                geo_features.append(pop_density_workday_weekend_diff)

                geo_features.append(reside)
                geo_features.append(life)
                geo_features.append(work)

                vertex_coords.append([cx, cy])
                vertex_features.append(geo_features)

                voronoi_geometry.Destroy()
                voronoi_geometry_intersect.Destroy()
                ######################################################################
                # # test
                # plt.fill(*zip(*polygon))
                # building_coords = get_coords_polygon(building_geometry)
                # plt.fill(*zip(*building_coords))
                # plt.show()
        elif len(centroid_points) > 0:
            for index in range(len(centroid_points)):
                # Calculate building area
                building = building_group_list[index]
                building_geometry = building.GetGeometryRef()
                building_geometry_buffer = building_geometry.Buffer(-0.5)
                building_area = building_geometry_buffer.GetArea()
                building_centroid_point = building_geometry_buffer.Centroid()
                cx, cy = building_centroid_point.GetX(0), building_centroid_point.GetY(0)
                # Calculate Impact Area Ratio (IMA)
                density = building_area / yard_geometry.GetArea()

                # Calculate Count Area Ratio
                count_density = len(building_group_list) / yard_geometry.GetArea()

                subobject = get_coords_polygon(building_geometry)
                ######################################################################
                geo_features = calculate_indicator(subobject)

                geo_features.append(density)

                geo_features.append(count_density)

                geo_features.append(day_workday_pop_density)
                geo_features.append(night_workday_pop_density)
                geo_features.append(day_workday_pop_density_rate)
                geo_features.append(night_workday_pop_density_rate)
                geo_features.append(day_weekend_pop_density_rate)
                geo_features.append(night_weekend_pop_density_rate)
                geo_features.append(pop_density_workday_weekend_diff)

                geo_features.append(reside)
                geo_features.append(life)
                geo_features.append(work)

                vertex_coords.append([cx, cy])
                vertex_features.append(geo_features)
                ######################################################################

        for building_feature_item in building_group_list:
            building_feature_item.Destroy()
        building_layer.ResetReading()
        yard_feature.Destroy()
        yard_feature = yard_layer.GetNextFeature()
        interpretedDic[gourp_id] = [label, vertex_coords, vertex_features]
        # print(gourp_id,label,vertex_coords)

    with open(output_json_file, 'w') as json_file:
        json.dump(interpretedDic, json_file, indent=2, ensure_ascii=False)
    building_data_store.Destroy()
    yard_data_store.Destroy()
    pop_workday_data_store.Destroy()
    pop_weekend_data_store.Destroy()
    # poi_data_store.Destroy()
    return

# Get field indexes with field name
def feature_field_index(field_name):
    index = None
    mapping = {"area": 0, "peri": 1, "long_chord": 2, "avg_radius": 3, "smbr_orien": 4, "edge_orien": 5, "bis_orien": 6, "wt_orien": 7,
    "ric_compa": 8, "ipq_compa": 9, "fra_compa": 10, "gib_compa": 11, "div_compa": 12, "elongation": 13, "ellip": 14, "concavity": 15,
    "dcm_index": 16, "bot_index": 17, "boy_index": 18, "m11": 19, "eccentric": 20, "density": 21, "count_density": 22,
    "pop_w_d": 23, "pop_w_n": 24, "pop_w_d_r": 25,
    "pop_w_n_r": 26, "pop_r_d_r": 27, "pop_r_n_r": 28, "pop_wr_dif": 29, "reside": 30, "publiclife": 31, "work": 32}
    if mapping.get(field_name) is not None:
        index = mapping[field_name]
    return index

# field indexes mapping
def feature_field_index_mapping():
    mapping = {"area": 0, "peri": 1, "long_chord": 2, "avg_radius": 3, "smbr_orien": 4, "edge_orien": 5, "bis_orien": 6, "wt_orien": 7,
    "ric_compa": 8, "ipq_compa": 9, "fra_compa": 10, "gib_compa": 11, "div_compa": 12, "elongation": 13, "ellip": 14, "concavity": 15,
    "dcm_index": 16, "bot_index": 17, "boy_index": 18, "m11": 19, "eccentric": 20, "density": 21, "count_density": 22, "pop_w_d": 23, "pop_w_n": 24, "pop_w_d_r": 25,
    "pop_w_n_r": 26, "pop_r_d_r": 27, "pop_r_n_r": 28, "pop_wr_dif": 29, "reside": 30, "publiclife": 31, "work": 32}
    return mapping

# Select the samples that meets the requirements based on the constraints, output json format
def select_features_json(result_file, output_file, fields_list, building_limit, building_limit_dic, label_limit, re_classify, re_classify_list):
    print("Loading the {} data".format(result_file))
    file = open(result_file, 'r', encoding='utf-8')
    dataset = json.load(file)
    file.close()

    field_index_list = []
    if "*" in fields_list:
        fields_mapping = feature_field_index_mapping()
        field_index_list = range(0, len(fields_mapping))
    else:
        for field_name in fields_list:
            index = feature_field_index(field_name)
            field_index_list.append(index)

    dataset_select = {}
    for k in dataset:
        # # 1 get the label(分类标签)、vertices coords（点坐标） and features(点特征) of this sample.
        [label, vertice_coords, vertice_features] = dataset[k]
        # print(len(vertice_features), len(vertice_coords))
        # building_limit_dic = {0: 7, 1: 6, 2: 4, 3: 6, 4: 7}

        if len(vertice_features) == 0:
            continue
        if building_limit is True:
            if len(vertice_coords) < building_limit_dic[label][0] or len(vertice_coords) > building_limit_dic[label][1]:
                continue

        # if building_limit is False:
        #     if len(vertice_coords) == 1:
        #         print(label, len(vertice_coords), k)
        # else:
        #     print(label, len(vertice_coords))
        print(label, len(vertice_coords))

        [reside, life, work] = \
            [vertice_features[0][30], vertice_features[0][31], vertice_features[0][32]]
        poi_heater_sort = [reside, life, work]

        max_heater_index = np.argmax(poi_heater_sort)
        # if max_heater_index != label:
        #     print("true label:" + str(label) + ", max_heater_index:" + str(max_heater_index))
        if label_limit is True:
            if max_heater_index != label:
                continue

        assert len(vertice_coords) == len(vertice_features)
        #"area"-0, "peri"-1, "long_chord"-2, "avg_radius"-3, "smbr_orien"-4, "edge_orien"-5, "bis_orien"-6, "wt_orien"-7,
        #"ric_compa"-8, "ipq_compa"-9, "fra_compa"-10, "gib_compa"-11, "div_compa"-12, "elongation"-13, "ellip"-14, "concavity"-15,
        #"dcm_index"-16, "bot_index"-17, "boy_index"-18, "m11"-19, "eccentric"-20, "density"-21, "pop_w_d"-22, "pop_w_n"-23, "pop_w_d_r"-24,
        #"pop_w_n_r"-25, "pop_r_d_r"-26, "pop_r_n_r"-27, "pop_wr_dif"-28, "reside"-29, "life"-30, "company"-31, "public"-32, "tour"-33
        vertice_features_select_list = []
        for i in range(0, len(vertice_features)):
            vertice_features_select = []
            for j in range(0, len(vertice_features[i])):
                if j in field_index_list:
                    vertice_features_select.append(vertice_features[i][j])
            vertice_features_select_list.append(vertice_features_select)
        if re_classify is True:
            label = re_classify_list[label]
        dataset_select[k] = [label, vertice_coords, vertice_features_select_list]

    with open(output_file, 'w') as json_file:
        json.dump(dataset_select, json_file, indent=2, ensure_ascii=False)

    return

# Calculate characteristic indicators
def test_calculate_indicators_json():

    building_file = "D:/PythonProject/function_area_identify_new/github/data/shp/building.shp"
    use_building_area_limit = True
    building_area_limit = 100.0
    use_yard_area_limit = True
    reside_yard_area_limit = 6000.0
    work_yard_area_limit = 4000.0
    yard_file = "D:/PythonProject/function_area_identify_new/github/data/shp/courtyard.shp"
    pop_workday_file = "D:/PythonProject/function_area_identify_new/github/data/shp/grid_workday_hour.shp"
    pop_weekend_file = "D:/PythonProject/function_area_identify_new/github/data/shp/grid_restday_hour.shp"
    poi_file = "D:/PythonProject/function_area_identify_new/github/data/shp/poi.shp"
    output_json_file = "D:/PythonProject/function_area_identify_new/github/data/json/samples.json"

    calculate_indicators_json(building_file, use_building_area_limit, building_area_limit, yard_file, use_yard_area_limit, reside_yard_area_limit, work_yard_area_limit, pop_workday_file, pop_weekend_file, poi_file, output_json_file)
    return

# Select samples
def test_select_features_json():
    # # for pca
    # result_file = "D:/PythonProject/function_area_identify_new/github/data/json/all_samples.json"
    # output_file = "D:/PythonProject/function_area_identify_new/github/data/json/samples_pca.json"
    # fields_list = ["*"]

    result_file = "D:/PythonProject/function_area_identify_new/github/data/json/all_samples.json"
    output_file = "D:/PythonProject/function_area_identify_new/github/data/json/samples.json"


    fields_list = {"area", "peri", "long_chord", "avg_radius", \
                   "smbr_orien", "edge_orien", "bis_orien", \
                   "ric_compa", "ipq_compa", "fra_compa", "gib_compa", "div_compa", \
                   "elongation", "concavity", \
                   "m11", "eccentric", \
                   "density", "count_density", \
                   "pop_w_d", "pop_w_n", "pop_w_d_r", "pop_w_n_r", "pop_r_d_r",  "pop_wr_dif", \
                   "reside", "publiclife", "work"}

    building_limit = True#必要的
    building_limit_dic = {0: [4, 50], 1: [2, 50], 2: [4, 50]}
    label_limit = False
    re_classify = False
    re_classify_list = {0: 0, 1: 1, 2: 2}
    select_features_json(result_file, output_file, fields_list, building_limit, building_limit_dic, label_limit, re_classify, re_classify_list)
    return

if __name__ == '__main__':
    # test_calculate_indicators_json()
    test_select_features_json()
