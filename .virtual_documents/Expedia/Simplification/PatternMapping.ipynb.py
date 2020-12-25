import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dis
from sklearn.linear_model import LinearRegression
from scipy.ndimage import filters
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt, seaborn
from io import StringIO
import cProfile
import os
import logging
import glob
import math
from graphviz import Source
plt.rcParams.update({'figure.max_open_warning': 0})
min_max_scaler = preprocessing.MinMaxScaler()
logger = logging.getLogger()
#     CRITICAL
#     ERROR
#     WARNING
#     INFO
#     DEBUG
logging.disable(logging.DEBUG);


# https://www.cs.princeton.edu/courses/archive/spring03/cs226/assignments/lines.html

def maxPoints(points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        def max_points_on_a_line_containing_point_i(i):
            """
            Compute the max number of points
            for a line containing point i.
            """
            def slope_coprime(x1, y1, x2, y2):
                """ to avoid the precision issue with the float/double number,
                    using a pair of co-prime numbers to represent the slope.
                """
                delta_x, delta_y = x1 - x2, y1 - y2
                if delta_x == 0:    # vertical line
                    return (0, 0)
                elif delta_y == 0:  # horizontal line
                    return (sys.maxsize, sys.maxsize)
                elif delta_x < 0:
                    # to have a consistent representation,
                    #   keep the delta_x always positive.
                    delta_x, delta_y = - delta_x, - delta_y
                gcd = math.gcd(round(delta_x), round(delta_y))
                slope = (round(delta_x) / gcd, round(delta_y) / gcd)
                slope_size = delta_y / delta_x
                return slope, slope_size


            def add_line(i, j, count, duplicates, slope_size, slope):
                """
                Add a line passing through i and j points.
                Update max number of points on a line containing point i.
                Update a number of duplicates of i point.
                """
                # rewrite points as coordinates
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[j][0]
                y2 = points[j][1]
                
                # add a duplicate point
                if x1 == x2 and y1 == y2:  
                    duplicates += 1
                # add a horisontal line : y = const
                elif y1 == y2:
                    nonlocal horizontal_lines
                    horizontal_lines += 1
                    count = max(horizontal_lines, count)
                # add a line : x = slope * y + c
                # only slope is needed for a hash-map
                # since we always start from the same point
                else:
                    slope_temp,slope_size_temp = slope_coprime(x1, y1, x2, y2)
                    lines[slope_temp] = lines.get(slope_temp, 1) + 1
                    if lines_dots.get(slope_temp) is None:
                        lines_dots[slope_temp] = set(((x1,y1),(x2,y2)))
                    else:
                        lines_dots[slope_temp].update([(x1,y1),(x2,y2)])
                    if lines[slope_temp] > count:
                        count = lines[slope_temp]
                        slope = slope_temp
                        slope_size = slope_size_temp
                return count, duplicates, slope_size, slope
            
            # init lines passing through point i
            lines, horizontal_lines = {}, 1
            # One starts with just one point on a line : point i.
            count = 1
            # There is no duplicates of a point i so far.
            duplicates = 0
            slope_size = 0
            slope = ()
            # Compute lines passing through point i (fixed)
            # and point j (interation).
            # Update in a loop the number of points on a line
            # and the number of duplicates of point i.
            for j in range(i + 1, n):
                count, duplicates, slope_size, slope = add_line(i, j, count, duplicates, slope_size, slope)
            return count + duplicates, slope_size, slope
            
        # If the number of points is less than 3
        # they are all on the same line.
        n = len(points)
        print("len(points):{}".format(n))
        if n < 3:
            return n
        
        intercept = 0
        max_count = 1
        # 存取同一斜率下的所有点
        lines_dots = {}
        # Compute in a loop a max number of points 
        # on a line containing point i.
        max_index = 0
        for i in range(n - 1):
            max_point_result = max_points_on_a_line_containing_point_i(i)
            if max_point_result[0] > max_count:
                max_count = max_point_result[0]
                slope_size = max_point_result[1]
                slope = max_point_result[2]
                max_index = i
        dots_len = len(lines_dots.get(slope))
        print("lines_dots.len:{}".format(dots_len))
        for iterm in lines_dots.get(slope):
            intercept+=iterm[1] - (iterm[0]*slope_size)
#         print(points[max_index])
#         intercept = points[max_index][1] - (slope[0]/slope[1]) * points[max_index][0]
#         return max_count, slope_size, intercept
        return max_count, slope_size, intercept/dots_len


HotelID = 16639
Observe = 'CostAmt'
GroupID = 1


#safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
 
# end of main

# HOME_FOLDER = '/Users/xyao/Library/Mobile Documents/com~apple~CloudDocs/JupyterHome/Simplification/'
HOME_FOLDER = './'
INPUT_FOLDER = './Data2/'
INPUT_FOLDER2 = './Result/MINE2/'
OUTPUT_FOLDER = './Result/MINE2/'

os.chdir(HOME_FOLDER)
read_data_rt = pd.read_csv('{}{}_{}_gp.csv'.format(INPUT_FOLDER2,HotelID,Observe), \
            encoding='utf-8', sep=',', engine='python', header=0).fillna(0)

read_data_rt = read_data_rt.loc[read_data_rt['GroupID']==GroupID]

#     RP2 = 260281795
#     RP1 = 260282228

read_data = pd.read_csv(INPUT_FOLDER+str(HotelID)+'_RatePlanLevelCostPrice.csv.zip', sep=',', engine='python', header=0).fillna(0)
read_data = read_data.loc[read_data['RatePlanID'].isin(read_data_rt['RatePlanID'])]

read_data = read_data.loc[(read_data['RatePlanLevel']==0) & (read_data['LengthOfStayDayCnt']==1) \
                        & (read_data['PersonCnt']==2)]

read_data = read_data[['StayDate',Observe,'RatePlanID']]

RP1 = read_data_rt['RatePlanID'].iloc[0]
rp1_dd = read_data.loc[read_data['RatePlanID']==RP1].set_index('StayDate')


rp_func = pd.DataFrame()
for i in range(1, len(read_data_rt.index)):
    logger.info("left compare length: {}".format(read_data_rt.size-i))
    RP2 = read_data_rt['RatePlanID'].iloc[i]

    rp2_dd = read_data.loc[read_data['RatePlanID']==RP2].set_index('StayDate')

    rp_ds = pd.merge(rp1_dd, rp2_dd, on='StayDate')
    # 删除 RatePlanID_x RatePlanID_y
    rp_ds_copy = rp_ds.copy(deep=True)
    rp_ds_copy = rp_ds_copy.drop(['RatePlanID_x','RatePlanID_y'], axis=1)
    max_count, slope, intercept = maxPoints(points = rp_ds_copy.values)
    print("max_count:{} slope:{} intercept:{}".format(max_count, slope, intercept))
#     print(rp_ds_copy.head(10))
    rp_ds_copy = preprocessing.StandardScaler().fit_transform(rp_ds_copy)
#     rp_ds_copy = min_max_scaler.fit_transform(rp_ds_copy)
    fit_X = rp_ds_copy[:,0].reshape((-1,1))
    fit_y = rp_ds_copy[:,1].reshape((-1,1))
#     print(fit_X)
#     X = rp_ds[Observe+'_x'].to_numpy().reshape(-1, 1)
#     y = rp_ds[Observe+'_y'].to_numpy().reshape(-1, 1)
#     x_minmax = MinMaxScaler.fit_transform(X)
    X = rp_ds[Observe+'_x'].to_numpy().reshape(-1, 1)
    y = rp_ds[Observe+'_y'].to_numpy().reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    pred_y = lr.predict(X)

    rp_func=rp_func.append([[RP2,'{:.4f}'.format(lr.score(X,y)),'{:.4f} * x {:+.4f}'.format(lr.coef_[0][0],lr.intercept_[0])]],ignore_index=True)
#     plt.xlim(X.min(), X.max())
#     plt.ylim(y.min(), y.max())
    fig, ax = plt.subplots(figsize=(18,7))
    
    ax.scatter(X, y,  color='blue')
    ax.plot(X, pred_y, color='green', linewidth=1)
    ax.plot(X, X*slope+intercept, color='r', linewidth=1, linestyle='--')
    rp_ds.to_csv('{}{}_Group{}_Line{}_{}_xy.csv'.format(OUTPUT_FOLDER,HotelID,GroupID,i,Observe), index=False)
    
    plt.show()
    plt.close()


rp_func.columns = ['RatePlanID','Accuracy','Formula']

rp_func.sort_values(by='Accuracy', ascending=False, inplace=True)

rp_func.to_csv('{}{}_{}_{}_func.csv'.format(OUTPUT_FOLDER,HotelID,GroupID,Observe), index=False)



