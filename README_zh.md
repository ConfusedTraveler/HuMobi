# HuMobi

<b> HuMobi 正在进行重大更新！如果您遇到任何问题，请放心，开发团队正在努力修复这些 bug。 </b>

您可以在 Google Colab 上找到本 README 的交互式版本：https://colab.research.google.com/drive/1nmYJDBYr1nN89LLx2NKP6Q5bqQ2mugbO?usp=sharing

## 目录
* [基本信息](#基本信息)
* [安装 HuMobi](#安装-humobi)
* [数据读取](#数据读取)
* [数据预处理](#数据预处理)
* [指标](#指标)
* [数据生成例程](#数据生成例程)
* [下一地点预测](#下一地点预测)
* [论文：通过模式匹配算法解释人类移动性预测](#论文通过模式匹配算法解释人类移动性预测)
* [已知问题](#已知问题)

## 基本信息

这是 HuMobi 库。它是一个专门用于人类移动性数据处理的 Python 库，主要扩展了 Pandas DataFrame 和 Geopandas GeoDataFrame，并便于对个体移动轨迹这一特定数据结构进行操作。下文将介绍如何在您的计算机上安装 HuMobi 库，以及一组覆盖该库大部分功能的演示示例。

该库主要用于处理个体移动轨迹，并专注于人类移动性预测与建模。最初，它由弗罗茨瓦夫环境与生命科学大学（Wrocław University of Environmental and Life Sciences）的博士生 Kamil Smolak 在其博士研究期间实现。

如果您使用了本库，请引用以下工作：
```
Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1-10.
```

这是一个不断扩展的项目，在您阅读本文时，新功能正在持续加入。目前，我正在实现人类移动性模型——这些模型尚未正常运行，也不在文档覆盖范围内。

HuMobi 库的当前功能包括：
* `TrajectoriesFrame` 基础类，用于加载和存储移动性数据。您可以在 `structures` 目录中找到该类；
* 个体和集体人类移动性统计指标；
* 实用的数据处理函数；
* 下一地点和下一时间窗口预测方法，包括马尔可夫链、深度学习和浅层学习模型；
* 数据聚合和过滤的预处理方法；
* 其他有用的数据处理工具。

## 安装 HuMobi

要安装 HuMobi 及其所有依赖，只需使用：

```
$ pip install HuMobi
```

由于特定库的组合，为该库设置合适的工作环境可能会比较棘手。因此，建议使用 virtualenv。该库的必需依赖包括：
* pandas >=1.1.5
* geopandas >=0.8.1
* tqdm >=4.59.0
* scipy >=1.5.2
* numpy >=0.19.5
* scikit-learn >=0.23.2
* biopython >=1.78
* shapely >=1.7.1
* numba >= 0.51.2
* tensorflow-gpu >= 2.4.1
* geofeather >=0.3.0

# 快速开始

下面是该库各种功能的演示示例。这些示例覆盖了 HuMobi 当前状态的大部分能力。请注意，未来还会进一步扩展——包括本文档。有关方法属性和函数的详细信息，请参见 `html` 文件夹中的文档。

某些方法和函数需要一些时间来执行。`tqdm` 库会启用进度条，用于估计剩余计算时间。

## 数据读取

数据的加载、存储和读取都在特殊的 `TrajectoriesFrame` 类中完成。这是一个基于 pandas DataFrame 的数据结构，具有 MultiIndex，由用户 id（上层）和时间戳（下层）组成。此外，它还包含一个 `geometry` 列，与 GeoPandas GeoDataFrame 的 geometry 列相同。

首先，导入必要的模块。我们将从 `humobi.structures` 导入 `trajectory` 模块，同时为了演示需要也导入 pandas。
```
from humobi.structures import trajectory as tr
import pandas as pd
```

`TrajectoriesFrame` 类位于 `trajectory` 模块中。`TrajectoriesFrame` 是一个智能类，可适应多种数据加载方式。例如，我们可以读取指定路径的文件：
```
# 从路径读取
in_path = """brightkite_sample.tsv"""  # 文件路径
df = tr.TrajectoriesFrame(in_path, {'names': ['id', 'datetime', 'lat', 'lon', 'place'], 'delimiter': '\t',
                                    'crs': 4326})  # 加载数据
```
如您所见：
* 第一个位置参数是文件路径。
* 除此之外，还可以传入 `pd.read_csv` 函数的 `**kwargs`。

此外，`TrajectoriesFrame` 接受两个元数据参数：
* `crs` - 根据 EPSG 分类的坐标参考系统编号
* `geom_cols` - 指示包含坐标的两个列。

请注意，当分隔符不是逗号时，提供 `delimiter` 关键字非常重要。
提供列名很有用，但 `TrajectoriesFrame` 会尝试自动判断哪一列是时间戳、哪一列是坐标。不过，为避免错误，请将时间戳列命名为 `time` 或 `datetime`，将几何列命名为 `lat` 和 `lon`。

该文件的前两行如下所示：
```
                                  id  ...                     geometry
user_id datetime                       ...                             
0       2010-10-17 01:48:53+00:00   0  ...  POINT (-104.99251 39.74765)
        2010-10-17 01:54:32+00:00   0  ...  POINT (-104.99251 39.74765)
```
> **_注意：_**  在某些 IDE 的调试模式下，您可以通过受保护的 `_base_class_view` 属性访问 TrajectoriesFrame 的 DataFrame 视图。

获取 `TrajectoriesFrame` 实例的另一种方式是将标准 DataFrame 转换为它：
```
# 从 DataFrame 转换
df = pd.read_csv(in_path, names=['id', 'datetime', 'lat', 'lon', 'place'], delimiter='\t')  # 使用 pd.read_csv 加载数据
df = tr.TrajectoriesFrame(df)  # 转换为 TRAJECTORIESFRAME
```

不带列名也可以读取，但不推荐：
```
# 无列名从 DataFrame 转换（不推荐）
df = pd.read_csv(in_path, delimiter='\t', header=None)  # 无列名信息加载
df = tr.TrajectoriesFrame(df)
```

此外，您也可以从 geofeather 文件读取：
```
# 从 GEOFEATHER 读取
df.from_geofeather("feather_path")
```

`TrajectoriesFrame` 可以使用以下方法保存为 CSV、geofeather 或 Shapefile：
```
df.to_csv("csv_path")
df.to_geofeather("feather_path")
df.to_shapefile("shape_path")
```

`TrajectoriesFrame` 还重写了数据空间转换方法 `to_crs`，可用于重新投影数据。只需调用：
```
df.to_crs(dest_crs = 3857, cur_crs = 4326)
```
这会将 TrajectoriesFrame 从 `EPSG:4326` 重新投影到 `EPSG:3857`。如果未提供 `cur_crs`，TrajectoriesFrame 将使用 `crs` 元数据。

> **_注意：_**  `TrajectoriesFrame` 基于 pandas DataFrame，因此可以在其上应用任何 pandas 函数——例如过滤和数据选择。

## 数据预处理

原始移动数据应经过预处理，以去除噪声、去除不重要的停留点并提取必要信息。HuMobi 库提供了数据预处理、分析和过滤的方法。该过程包括两个步骤。首先，去除噪声数据并检测停留位置。第二步，将停留位置聚合为停留区域（stay-regions），并转换为移动序列。有关方法论的详细信息，请参阅以下出版物：
```
Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1-10.
```

数据预处理工具位于 `preprocessing` 模块中。一些统计和数据压缩方法位于 `tools` 模块中。
首先，导入必要的函数并读取我们在上一小节中使用 `to_csv` 方法保存的数据示例。
```
from humobi.structures import trajectory as tr
from humobi.preprocessing.filters import stop_detection
from humobi.tools.user_statistics import *
from humobi.tools.processing import start_end

in_path = """converted_sample.csv"""
df_sel = tr.TrajectoriesFrame(in_path, {'crs': 4326})  # 已转换 - 包含 GEOMETRY 和 MULTIINDEX，保存为 CSV（参见 data_reading.py 演示）
geom_cols = df_sel.geom_cols  # 存储几何列
crs = df_sel.crs  # 存储 CRS
```

### 数据（用户）选择

首先，介绍如何选择特定的移动轨迹，以便后续过滤数据。`TrajectoriesFrame` 提供了 `uloc` 方法，允许您使用用户 id 选择一个或多个用户。例如，选择 id 为 `0` 的用户：
```
one_user = df_sel.uloc(0)
```

要获取所有用户 id 的列表，请使用 `get_users()` 方法：
```
users_list = df_sel.get_users()  # 所有 ID 的列表
```

然后，将该列表传递给 `uloc` 将选择数据中的所有用户（因此结果不会改变）：
```
many_users = df_sel.uloc(users_list)
```

您也可以使用标准的 `loc` 和 `iloc` pandas 命令。

### 用户统计

许多数据选择方法基于选择具有某些全局统计特征的用户，例如数据完整性或轨迹持续时间。`tools.user_statistics` 模块包含一些可用于计算这些指标的统计量。所有结果都返回为以用户 id 为索引、统计量为值的 pandas `Series`。可用统计量包括：

#### 空记录比例
空记录比例以给定时间分辨率表示。这是全局计算的——您可以使用选择方法限制时间范围，或者与 `pd.rolling` 一起使用以获得移动值。
```
frac = fraction_of_empty_records(df_sel, resolution='1H')  # 缺失记录比例
```

#### 记录总数
记录的总数：
```
count = count_records(df_sel)  # 记录总数
```

#### 每时间段记录总数
按时间段计算的记录总数：
```
count_per_time_frame = count_records_per_time_frame(df_sel, resolution='1D')  # 每时间段记录总数
```

#### 轨迹持续时间
轨迹的总长度，以时间单位表示。`count_empty` 决定是否应考虑空记录或将其排除。如果排除，该值将按空记录时间段的数量减少。
```
trajectories_duration = user_trajectories_duration(df_sel, resolution='1H', count_empty=False)  # 轨迹总长度
```

#### 连续记录数
以给定时间单位表示的最高连续记录数。
```
consecutive = consecutive_record(df_sel, resolution='1H')
```

#### 使用统计量过滤
现在，介绍如何使用这些统计量来过滤数据。例如，我们想选择空记录比例低于 90%、轨迹长度超过 6 天且至少有 100 条数据的用户。我们将使用集合交集来找到满足这三个条件的所有用户。
```
# 使用用户统计量过滤
frac = fraction_of_empty_records(df_sel, '1H')
level1 = set(frac[frac < 0.9].index)  # 缺失记录比例 < 0.9

traj_dur = user_trajectories_duration(df_sel, '1D')
level2 = set(traj_dur[traj_dur > 6].index)  # 数据超过 6 天

counted_records = count_records(df_sel)
level3 = set(counted_records[counted_records >= 100].index)  # 总记录数至少 100

# 索引选择
selection = level1.intersection(level2)
selection = selection.intersection(level3)
df_sel = df_sel.uloc(list(selection))  # 使用 ULOC 方法进行用户过滤

# 按时间戳排序（以防万一）
df_sel = df_sel.sort_index(level=1)

# 重新读取结构
df_sel = tr.TrajectoriesFrame(df_sel, {'crs': crs, 'geom_cols': geom_cols})
```

### 停留点检测

停留点检测算法使用简单。`preprocessing.filters.stop_detection` 函数可以快速检测停留点：
```
stops = stop_detection(df_sel, distance_condition=300, time_condition='10min')
```
有关算法细节，请参阅上述出版物。有两个可调参数：
* `distance_condition`（此处为 300 米）- 始终以米为单位
* `time_condition`（此处为 10 分钟）。

该函数会临时将数据转换为 `EPSG:3857`。请注意，它使用多线程，因此多核处理器会更有帮助。

`stop_detection` 函数会添加一个新的布尔列 `is_stop`。您可以使用以下方式仅过滤停留点：
```
df_sel = stops[stops['is_stop'] == True]
```

最好也删除重复项，因为有时可能会产生重复：
```
df_sel = df_sel.drop_duplicates()
```

此外，为减小数据大小，我们可以通过添加每个位置访问的 `start` 和 `end` 时间，将停留点压缩为单行数据。调用：
```
df_sel = start_end(df_sel)
```
最终，生成的 `TrajectoriesFrame` 将如下所示：
```
                                 id        lat         lon                                     place                     geometry  is_stop                      date                     start                       end
user_id datetime                                                                                                                                                                                                         
0       2009-05-29 00:04:23+00:00   0  39.759608 -104.984862          6346d66a3aa011de83f8003048c0801e  POINT (-104.98486 39.75961)     True 2009-05-29 00:04:23+00:00 2009-05-29 00:04:23+00:00 2009-05-29 02:29:20+00:00
        2009-05-30 02:12:30+00:00   0  39.890648 -105.068872          dd7cd3d264c2d063832db506fba8bf79  POINT (-105.06887 39.89065)     True 2009-05-30 02:12:30+00:00 2009-05-30 02:12:30+00:00 2009-05-30 07:28:16+00:00
```

### 数据聚合

在停留点检测之后，数据最终可以通过空间（停留区域检测）和时间聚合转换为移动序列。

停留区域检测可以使用多种方法完成，最常用的是基于网格的方法或聚类方法。此外，时间聚合有两种方法：下一时间窗口（next time-bin）和下一地点（next place）。下面介绍如何将数据转换为各种移动序列。首先，导入必要的函数。
```
from humobi.structures import trajectory as tr
from humobi.preprocessing.temporal_aggregation import TemporalAggregator
from humobi.preprocessing.spatial_aggregation import GridAggregation, ClusteringAggregator, LayerAggregator
from humobi.preprocessing.filters import next_location_sequence
```

#### 空间聚合

空间聚合（停留区域检测）应首先进行。此过程将添加一个新的 `labels` 列，用于标识唯一的停留区域。

> **__注意：__** 如果您已经有聚合数据，只想为每个几何添加唯一标签，请使用 `humobi.misc.utils` 模块中的 `to_labels()` 函数。

`humobi.preprocessing.spatial_aggregation` 模块提供以下空间聚合类：
* `GridAggregator`
* `ClusteringAggregator`
* `LayerAggregator`

要执行聚合，首先需要定义一个聚合器类。创建聚合器时，首先传入数据和控制聚合行为的参数。然后，可以调用 `aggregate` 方法执行数据聚合。

##### 网格聚合器（Grid Aggregator）

`GridAggregator` 可将数据快速聚合到指定分辨率的规则网格中。有一些已实现的行为。例如，您只需传入 `resolution` 参数，网格就会适配到数据范围。
```
gird_resolution = 1000  # 定义空间单元（网格）
grid_agg = GridAggregation(gird_resolution)  # 定义网格聚合算法
df_sel_grid = grid_agg.aggregate(df_sel, parallel=False)  # 调用聚合
```
* 当您想自己设置网格范围时，可以传入 `x_min`、`x_max`、`y_min`、`y_max` 参数来设置聚合网格的范围。
* 您可以传入 `origin` 参数来指定网格是否应以数据为中心。
* `aggregate()` 方法的 `parallel` 参数会调用多线程处理，但由于开销问题，这不一定比单核方法更快。

##### 聚类聚合器（Clustering Aggregator）

`ClusteringAggregator` 允许您传入任何 scikit-learn 聚类算法来执行停留点聚类。在下面的示例中，我们使用 `DBSCAN` 类进行聚类：
```
from sklearn.cluster import DBSCAN
eps = 300  # 定义空间单元
min_pts = 2  # 其他超参数
clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps, "min_samples": min_pts})  # 定义空间聚合算法
df_sel_dbscan = clust_agg.aggregate(df_sel)  # 调用空间聚合
```
如您所见，聚类方法的所有参数都可以作为 `**kwargs` 传入。该类使用 scikit-learn 库内实现的多线程。

##### 图层聚合器（Layer Aggregator）

第三个类是 `LayerAggregator`。该类使用外部文件执行聚合。其功能基于 GeoPandas 的空间连接函数。只需调用：
```
layer_agg = LayerAggregator('path_to_outer_layer',**kwargs)
df_sel_layer = layer_agg.aggregate(df_sel)
```

#### 时间聚合

时间聚合功能位于 `humobi.preprocessing.temporal_aggregation` 模块中，该模块包含 `TemporalAggregator` 类。时间聚合有两种方法：下一时间窗口和下一地点。

##### 下一时间窗口（Next time-bin）
下一时间窗口方法将访问过的地点序列转换为等间隔的时间窗口。要执行下一时间窗口聚合，只需实例化 TemporalAggregator 类并传入用于聚合的时间单位：
```
time_unit = '1H'  # 定义时间单元
time_agg = TemporalAggregator(time_unit)  # 定义时间聚合算法
```
在上面的示例中，我们选择时间窗口为每小时分辨率。现在我们可以调用数据的聚合：
```
df_sel_dbscan_time = time_agg.aggregate(df_sel_dbscan, parallel=True)
```
上面的代码行将根据文献中介绍的方法执行时间窗口聚合。可能出现三种情况：
* 该时间窗口没有数据 -> 在这种情况下，时间窗口将为空
* 一个时间窗口内访问了多个停留区域 -> 选择用户停留时间最长的停留区域
* 一个时间窗口内访问了多个停留区域且用户在这些区域停留时间相同 -> 选择用户过去停留时间最长的区域

* 时间聚合可以使用 `parallel` 设置运行，这比其单核版本稍快。
* `aggregate()` 方法有一个 `fill_method` 参数，可以设置为 `ffill` 或 `bfill` 来填充缺失数据，或者设置 `drop_empty` 参数来移除缺失的时间窗口。

时间聚合计算量较大，可能需要一些时间。

> **_注意：_** 时间窗口总是从午夜开始。

##### 下一地点（Next place）

在下一地点方法中，会移除在同一停留区域的所有连续访问记录。这可以使用 `humobi.preprocessing.filters` 模块中的 `next_location_sequence` 函数完成。
```
df_sel_time_seq = next_location_sequence(df_sel_dbscan_time)  # 创建下一地点序列
```

## 指标

处理后，可以在移动序列上计算各种指标。我们将它们分为个体指标和集体指标。个体指标针对 `TrajectoriesFrame` 中的每个 id 分别计算。集体指标以分布形式呈现，或指代停留区域。

首先，导入所有指标：
```
from humobi.structures import trajectory as tr
from humobi.measures.individual import *
from humobi.measures.collective import *
```

我们假设处理后的数据存储在 `df_sel_dbscan_time` 变量中。

### 个体指标

#### 不同地点数量

该指标计算每个个体访问的不同地点数量。要计算它，请调用 `humobi.measures.individual` 模块中的 `num_of_distinct_locations()` 函数。
```
distinct_total = num_of_distinct_locations(df_sel_dbscan_time)
```

#### 访问频率

该指标计算用户访问每个停留区域的频率。执行：
```
vfreq = visitation_frequency(df_sel_dbscan_time)
```

#### 随时间变化的不同地点数量

这是不同地点数量指标的变体，计算从移动轨迹开始每个时间步访问的不同地点数量。该函数需要两个额外参数。
* `resolution` 确定时间步的大小。
* `reaggregate` 是一个布尔参数，如果需要，将运行 TemporalAggregator 将数据转换为新的时间窗口大小。

执行：
```
distinct_over_time = distinct_locations_over_time(df_sel_dbscan_time, resolution='1H', reaggregate=False)
```

#### 跳跃长度（Jump lengths）
该函数计算移动序列中各地点之间所有行程的长度。
```
jump = jump_lengths(df_sel_dbscan_time)
```

#### 非零行程（Nonzero trips）
该函数计算所有行程的数量（距离 > 0 的行程）
```
trips = nonzero_trips(df_sel_dbscan_time)
```

#### 自转移（Self-transition）
该函数计算用户在下一个时间窗口停留在同一位置的情况数量（在下一地点方法中，该值将始终等于 0）。
```
st = self_transitions(df_sel_dbscan_time)
```

#### 等待时间（Waiting times）
该函数计算 `TrajectoriesFrame` 中每次转移的等待时间。
* 该函数需要指定 `time_unit`，用于控制等待时间的表达单位。
```
wt = waiting_times(df_sel_dbscan_time, time_unit = 'h')
```

#### 质心（Center of mass）
计算每个用户轨迹的质心。
```
mc = center_of_mass(df_sel_dbscan_time)
```

#### 回转半径（Radius of gyration）
计算每个用户的回转半径。
* 可选地，可以使用 `time_evolution` 参数来表示该指标随时间的演变。
```
rog = radius_of_gyration(df_sel_dbscan_time, time_evolution=False)
rog_time = radius_of_gyration(df_sel_dbscan_time, time_evolution=True)
```

#### 均方位移（Mean square displacement）
计算每个用户的均方位移（MSD）。
* 可选地，可以使用 `time_evolution` 参数来表示该指标随时间的演变。
```
msd = mean_square_displacement(df_sel_dbscan_time, time_evolution=False)
msd_time = mean_square_displacement(df_sel_dbscan_time, time_evolution=True)
```
* `from_center` 参数可用于计算相对于轨迹质心的 MSD（如果为 False，则从第一个点计算）。
* `reference_locs` 可以传入以确定每个人的自定义参考点。这必须是一个以用户 id 为索引、以点几何为值的 Series。

#### 返回时间（Return time）
计算每个用户轨迹中每个唯一地点的返回时间。
* `time_unit` 指定返回时间的表达单位。
```
rt = return_time(df_sel_dbscan_time, time_unit = 'H')
```
可选地，该指标可以按地点计算，表示任何人平均返回该地点所需的时间。这将生成一个 DataFrame，包含返回次数和平均返回时间。
```
rt_place = return_time(df_sel_dbscan_time, by_place=True)
```

#### 随机熵与可预测性
使用 `Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. Science, 327(5968), 1018–1021. https://doi.org/10.1126/science.1177170` 中定义的方程，计算 TrajectoriesFrame 中每个用户的随机熵。类似地，可预测性使用上述论文中的熵和 Fano 不等式计算。
```
ran_ent = random_entropy(df_sel_dbscan_time)
random_pred = random_predictability(df_sel_dbscan_time)
```
> **__注意：__** `random_predictability` 以 DataFrame 形式返回熵和可预测性。

#### 无相关熵与可预测性
使用 `Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. Science, 327(5968), 1018–1021. https://doi.org/10.1126/science.1177170` 中定义的方程，计算 TrajectoriesFrame 中每个用户的无相关熵。类似地，可预测性使用上述论文中的熵和 Fano 不等式计算。
```
unc_ent = unc_entropy(df_sel_dbscan_time)
unc_pred = unc_predictability(df_sel_dbscan_time)
```
> **__注意：__** `unc_predictability` 以 DataFrame 形式返回熵和可预测性。

#### 真实熵与可预测性
使用 Lempel-Ziv 压缩算法计算 TrajectoriesFrame 中每个用户的真实熵，方法定义见
`Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. Science, 327(5968), 1018–1021. https://doi.org/10.1126/science.1177170`，并根据
`Xu, P., Yin, L., Yue, Z., & Zhou, T. (2019). On predictability of time series. Physica A: Statistical Mechanics and Its Applications, 523, 345–351. https://doi.org/10.1016/j.physa.2019.02.006`
和
`Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1–10. https://doi.org/10.1038/s41598-021-94102-x`
的研究结果进行了修正。
此外，当缺失数据比例超过 15% 时，使用 `Ikanovic, E. L., & Mollgaard, A. (2017). An alternative approach to the limits of predictability in human mobility. EPJ Data Science, 6(1). https://doi.org/10.1140/epjds/s13688-017-0107-7` 的方法估计熵。

可预测性使用 Song 等人（2010）论文中的熵和 Fano 不等式计算。

> **__注意：__** 当缺失数据比例 >90% 时，__无法__ 计算真实熵。

> **__注意：__** 这些函数使用 GPU 执行计算。请确保您的机器上配置了 CUDA。CPU 版本不可用，因为 CPU 上的计算执行时间长得不合理。
```
real_ent = real_entropy(df_sel_dbscan_time)
real_pred = real_predictability(df_sel_dbscan_time)
```
> **__注意：__** `real_predictability` 以 DataFrame 形式返回熵和可预测性。

#### 平稳性（Stationarity）
根据 Teixeira 等人（2019）的研究，将平稳性计算为地点中的平均停留长度。详见 `Teixeira, D., Almeida, J., Viana, A. C., Teixeira, D., Almeida, J., Carneiro, A., … Viana, A. C. (2021). Understanding routine impact on the predictability estimation of human mobility To cite this version : HAL Id : hal-03128624 Understanding routine impact on the predictability estimation of human mobility.`。

```
stat = stationarity(df_sel_dbscan_time)
```

#### 规律性（Regularity）
根据 Teixeira 等人（2019）的研究，将规律性计算为序列长度与唯一符号数量的比值。详见 `Teixeira, D., Almeida, J., Viana, A. C., Teixeira, D., Almeida, J., Carneiro, A., … Viana, A. C. (2021). Understanding routine impact on the predictability estimation of human mobility To cite this version : HAL Id : hal-03128624 Understanding routine impact on the predictability estimation of human mobility.`。

```
regul = regularity(df_sel_dbscan_time)
```

### 集体指标

#### 出行距离分布
计算每个用户的出行距离分布。可以定义两个互斥参数：
* `bin_size` - 确定输出直方图的柱宽，
* `n_bins` - 确定输出直方图的柱数。（默认 = 20）
```
dist = dist_travelling_distance(df_sel_dbscan_time)
```

#### 流的对成对比较
计算每个聚合单元的流数量。
* 使用 `flows_type` - 可以统计 `all`（所有）流、仅 `incoming`（流入）流或仅 `outgoing`（流出）流。
```
pairwise_flows = flows(df_sel_dbscan_time, flows_type='all')
```

## 数据生成例程

生成合成数据对于验证具有已知统计性质的序列上的算法和假设可能很有用。`humobi.misc.generators` 模块中提供了几种生成例程。输出是一个带有 `labels` 列的 DataFrame，用于标识唯一地点：
```
                             labels
user_id datetime                                                      
0       1970-01-01 00:00:00       0 
        1970-01-01 01:00:00       0
```

每个例程都有可用参数：
* `users` - 要生成的唯一序列数量
* `places` - 生成所用词汇表的大小
这些参数可以传入固定值或值列表。如果使用后者，每个序列将使用从列表中随机选取的值生成。某些例程还有更多参数。详见下文。

导入模块：
```
from humobi.misc.generators import *
```

### 随机序列

随机序列，每个符号从可用词汇表中随机生成。可以确定额外的 `length` 参数。它可以是单个值或要从中随机选取的值列表。
```
random_seq = random_sequences_generator(users=10, places=10, length=100)
```

### 确定性序列

确定性序列遵循一系列递增符号，直到词汇表大小。到达词汇表末尾后，序列重复。例如，当词汇表大小为 4 时，序列将遵循 `[0, 1, 2, 3, 0, 1, 2, 3, 1, ...]` 的例程。可以确定额外的 `repeats` 参数。它可以是单个值或要从中随机选取的值列表。
```
deter_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
```

### 马尔可夫序列

马尔可夫序列遵循确定性序列，但在每一步以概率 `prob` 插入一个随机符号。
```
markovian_seq = markovian_sequences_generator(users=10, places=10, length=500, prob=.3)
```

### 探索性序列

探索性序列生成一系列唯一、不重复的符号。

```
ex_seq = exploratory_sequences_generator(users=10, places=10)
```

### 自转移序列

自转移序列类似于确定性序列，但每个符号在进入下一个之前会重复多次。每个自转移重复的次数由符号数量和序列长度决定。
```
st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
```

### 非平稳序列

非平稳序列使用 `states` 生成符号，每个状态都有其符号生成例程。每个状态的概率在过程开始时随机分配。
```
non_st_seq = non_stationary_sequences_generator(users=10, places=10, states=5, length=100)
```

## 下一地点预测

最后，让我们做一些预测。有多种算法可用，包括：
* 朴素方法
* 马尔可夫链
* 浅层学习分类算法
* 深度学习网络

首先，导入必要的模块：
```
from humobi.predictors.wrapper import *
from humobi.predictors.deep import *
```
> **__注意：__** 该模块正在开发中，结构中仍存在一些不一致之处。未来会进行调整，以便于创建管道。

### 数据划分

要进行预测，我们需要将数据划分为训练集和测试集。我们将使用 `humobi.predictors.wrapper` 模块中的 `Splitter` 类。该类允许一次性划分整个 TrajectoriesFrame，通过由 `split_ratio` 参数（表示测试集大小）确定的比例划分每个用户的轨迹。
例如，划分生成的马尔可夫序列：
```
data_splitter = Splitter(markovian_seq, split_ratio=.2)
```
划分后的数据可作为 `split` 实例属性访问。
* `split.data` 提供对输入数据的访问
* `split.cv_data` 产生训练-验证集对的列表。可以向 `Splitter` 传入额外的 `n_splits` 参数，使用 KFold 方法生成更多训练-验证数据集对。每对包括训练集特征（索引 0）、训练集标签（索引 1）、验证集特征（索引 2）和验证集标签（索引 3）。
* `split.test_frame_X` 提供对测试集特征的访问
* `split.text_frame_Y` 提供对测试集标签的访问

* 可以在 `Splitter` 类上设置 `horizon` 参数。如果 `horizon > 1`，所有生成的特征数据集（X 集）将包含额外的列，其中包含直到 horizon 长度的所有先前时间步。此外，训练/验证集和测试集将与 `horizon` 大小重叠。

让我们将数据集赋值给一些变量：
```
test_frame_X = data_splitter.test_frame_X
test_frame_Y = data_splitter.test_frame_Y
cv_data = data_splitter.cv_data
```

数据划分完成后，我们就可以进行预测了。

### TopLoc

TopLoc 算法是一种朴素算法，改编自 `Cuttone, A., Lehmann, S., & González, M. C. (2018). Understanding predictability and exploration in human mobility. EPJ Data Science, 7(1). https://doi.org/10.1140/epjds/s13688-017-0129-1`。进行预测时，它假设下一个访问地点是训练集中访问次数最多的地点。`TopLoc()` 类位于 `humobi.predictors.wrapper` 模块中。要在数据集上执行预测，请调用：
```
TopLoc(train_data=cv_data, test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()
```
调用 `predict()` 将返回预测结果（DataFrame，其中第一列是测试集标签，第二列是预测值）和准确率得分。

### MarkovChain

可以通过包装函数 `markov_wrapper` 调用 N 阶马尔可夫链。该包装器一次接受整个数据集，并在内部执行数据划分。测试-训练比例由 `test_size` 参数定义。马尔可夫链的阶数由 `state_size` 参数定义。例如，要调用 2 阶马尔可夫链，请执行以下行：
```
MC2 = markov_wrapper(markovian_seq, test_size=.2, state_size=2)
```
此外，该函数接受两个参数：`update` - 如果为 True，则每次预测符号后都会重建链；`online` - 如果为 True，则允许算法在预测时查看测试集中的最后 `state_size` 个符号。该函数返回预测结果和准确率得分。

### 浅层学习方法

可以使用 scikit-learn 库的分类方法进行预测。`humobi.predictors.wrapper` 模块允许使用 `SKLearnPred()` 类应用任何算法进行下一地点预测。为此，首先导入一个预测算法：
```
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier
```
然后我们按以下方式实例化 `SKLearnPred` 类：
```
predic = SKLearnPred(algorithm=clf, training_data=split.cv_data, test_data=[split.test_frame_X, split.test_frame_Y], param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]}, search_size=5, cv_size=5, parallel=False)
```
* 算法必须作为 `algorithm` 参数传入。
* `training_data` 和 `test_data` 必须从 `Splitter()` 类传入。
* `param_dist` 允许传入预测算法候选超参数的字典。这些将在带交叉验证的 RandomizedSearch 期间进行测试，以找到最佳解决方案。
* 交叉验证的次数由 `cv_size` 参数决定。
* 测试组合的数量由 `search_size` 参数决定。目前 `parallel` 计算尚不可行。

要执行预测，首先需要训练预测算法。调用
```
predic.learn()
```
将训练一组算法，每个数据集中的每条轨迹一个。然后调用
```
predic.test()
```
将在测试集上执行预测。准确率可以通过 `predic.scores` 属性访问，预测结果可以通过 `predic.predictions` 属性访问。

### 深度学习 methods

目前，HuMobi 库提供两种用于移动性预测的网络架构。未来会进一步扩展。它们是带嵌入层和 dropout 的单层和双层门控循环单元（GRU）网络，可在 `humobi.predictors.deep` 模块中访问。

要使用深度学习网络进行预测，首先调用 `DeepPred` 类。该类在初始化时接受以下参数：
* `model` - 使用单层 GRU 请调用 `'GRU'`，使用双层 GRU 请调用 `'GRU2'`
* `data` - TrajectoriesFrame（不是来自 Splitter）
* `test_size` - 测试集大小
* `folds` - 交叉验证的折数
* `batch_size` - 网络的批次大小。
* `embedding_dim` - 网络第一层拟合的嵌入向量的维度
* `rnn_units` - GRU 层中的神经元数量

使用示例：
```
GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=5, window_size=5, batch_size=1, embedding_dim=512, rnn_units=1024)
```
要执行预测，请调用：
```
GRU.learn_predict()
```
要通过 `GRU.predictions` 访问预测结果，通过 `GRU.scores` 访问准确率得分。

## 论文：通过模式匹配算法解释人类移动性预测

由 `Kamil Smolak, Witold Rohm, and Katarzyna Siła-Nowicka` 发表的论文 `Explaining human mobility predictions through a pattern matching algorithm` 提出并评估了五个新指标，用于衡量移动性数据的实际可预测性，并解释实际预测准确率的变异性。这些指标是：
* 密集重复性（Dense Repeatability, DR）
* 稀疏重复性（Sparse Repeatability, SR）
* 等稀疏重复性（Equally Sparse Repeatability, ESR）
* 全局对齐（Global Alignment, GA）
* 迭代全局对齐（Iterative Global Alignment, IGA）

这些指标已在 HuMobi 库中实现，可在 `humobi.measures.individual` 模块中使用。除了上述指标外，该论文中的所有工作都是使用本库完成的。这包括数据过滤和预处理（参见 [数据预处理](#数据预处理)）、部分计算的指标（特别是真实可预测性、平稳性和规律性，详见 [指标](#指标)）。预测及其准确率按照 [下一地点预测](#下一地点预测) 部分所示完成。

这些指标基于训练集和测试集。可以通过 `Splitter()` 获得：
```
X = pd.concat([split.cv_data[0][0], split.cv_data[0][2]])
Y = split.test_frame_X
```

上述任一指标都可以通过传入 `train_frame` 和 `test_frame` 参数简单计算：
```
DR = repeatability_dense(train_frame=X, test_frame=Y)
SR = repeatability_sparse(train_frame=X, test_frame=Y)
ESR = repeatability_equally_sparse(train_frame=X, test_frame=Y)
GA = global_alignment(train_frame=X, test_frame=Y)
IGA = iterative_global_alignment(train_frame=X, test_frame=Y)
```
这将返回一个 pd.Series，包含为数据中每个用户计算的值。

## 已知问题

其中一个已知问题与 crs 元数据设置有关。可能会发生向 `TrajectoriesFrame` 传入 `crs` 参数后未将其分配给数据集的情况。这是 GeoPandas 库和 Anaconda 的已知问题。正确设置 proj4 库可解决此问题。

根据此处 Dorregaray 的回答：[StackOverflow 问题](#https://stackoverflow.com/questions/55390492/runtimeerror-bno-arguments-in-initialization-list/58009620#58009620)，解决方案是编辑位于 `...\Anaconda3\Lib\site-packages\pyproj` 的 `datadir.py` 文件中的路径。应将其重命名为 `.../Anaconda3/Library/share`。
