import os
import pandas as pd
from datetime import datetime, timedelta
from tkinter import Tk, filedialog, messagebox, Label, StringVar, Text, Scrollbar, VERTICAL, END
import tkinter as tk
from tkinter.ttk import Progressbar
import subprocess
import numpy as np
import geopandas as gpd
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import LineString, Polygon, MultiPolygon
from osgeo import gdal, ogr, osr
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class RedirectText:
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.insert(END, string)
        self.output.see(END)

    def flush(self):
        pass

# 진행바 업데이트
def update_progress(progress_bar, progress_var, current_step, total_steps):
    progress_percentage = min((current_step / total_steps) * 100, 100)
    progress_var.set(f"{progress_percentage:.2f}%")
    progress_bar['value'] = progress_percentage
    progress_bar.update_idletasks()

# 중앙좌표+K-River결과로드창
def select_file(title):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

def create_input_files(progress_var, progress_bar):
    input_file_path = select_file('Select center.txt file')
    if not input_file_path:
        messagebox.showinfo("No file selected", "center.txt 파일을 선택하지 않았습니다.")
        return

    result_file_path = select_file('Select result_K-River_WL.csv file')
    if not result_file_path:
        messagebox.showinfo("No file selected", "result_K-River_WL.csv 파일을 선택하지 않았습니다.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_by_river = {}

    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5:
            main_river_name = parts[0].replace(',', '').strip()
            branch_name = parts[1].strip()
            full_river_name = f"{main_river_name}_{branch_name}"
            section_number = parts[2].replace(',', '').strip()
            section = f"{main_river_name}-{branch_name}-{float(section_number):.4f}".replace(',,', '-').replace(',', '')
            x = parts[3].strip()
            y = parts[4].strip()

            if full_river_name not in data_by_river:
                data_by_river[full_river_name] = []
            data_by_river[full_river_name].append([section, x, y])

    wl_data = pd.read_csv(result_file_path)

    total_steps = len(data_by_river) * len(wl_data)
    current_step = 0

    for full_river_name, data in data_by_river.items():
        river_dir = os.path.join(script_dir, 'FIM_input_data', full_river_name)
        os.makedirs(river_dir, exist_ok=True)

        river_df = pd.DataFrame(data, columns=['section', 'x', 'y'])

        for time_idx in range(len(wl_data)):
            time = wl_data.iloc[time_idx, 0]
            time_formatted = datetime.strptime(time, '%Y-%m-%d %H:%M:00').strftime('%Y%m%d%H%M')
            output_file_path = os.path.join(river_dir, f"{full_river_name}_{time_formatted}.csv")

            river_df_with_wl = river_df.copy()
            river_df_with_wl['WL'] = None

            for idx, section in enumerate(river_df_with_wl['section']):
                if section in wl_data.columns:
                    river_df_with_wl.at[idx, 'WL'] = wl_data.at[time_idx, section]
                else:
                    print(f"Section {section} not found in wl_data columns.")

            def extract_numeric(section):
                parts = section.split('-', 2)
                num_part = float(parts[-1])
                return num_part

            river_df_with_wl['section_numeric'] = river_df_with_wl['section'].apply(extract_numeric)
            river_df_with_wl = river_df_with_wl.sort_values(by='section_numeric', ascending=False).drop(columns='section_numeric')

            river_df_with_wl.to_csv(output_file_path, index=False, encoding='utf-8')
            print(f"Data successfully saved to {output_file_path}")

            # 진행률 업데이트
            current_step += 1
            if current_step > total_steps:
                current_step = total_steps
            update_progress(progress_bar, progress_var, current_step, total_steps)

    print(f"Input_file_처리완료!")
    messagebox.showinfo("Success", "Input files created successfully!")

def setup_environment():
    os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'
    os.environ['PROJ_LIB'] = os.path.join(sys.exec_prefix, 'Library', 'share', 'proj')
    gdal_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
    if os.name == 'nt' and sys.version_info >= (3, 8):
        os.add_dll_directory(gdal_dll_path)

def reproject_raster(src_path, dst_path, dst_crs):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        logging.error(f"Cannot open file {src_path}.")
        sys.exit(1)
    dst_ds = gdal.Warp(dst_path, src_ds, dstSRS=f'EPSG:{dst_crs}')
    if dst_ds is None:
        logging.error(f"Failed to reproject file {src_path}.")
        sys.exit(1)
    dst_ds.FlushCache()
    dst_ds = None

def process_file(file_path, buffer_distance, dem_file_path):
    df = pd.read_csv(file_path, skiprows=1, sep=',', names=['section', 'x', 'y', 'WL'])
    points = df.drop(['section', 'WL'], axis=1).to_numpy()

    line = LineString(points)
    line_buffer = line.buffer(buffer_distance, cap_style=2)

    boundary_coords = []
    if hasattr(line_buffer.boundary, 'geoms'):
        for geom in line_buffer.boundary.geoms:
            boundary_coords.extend(list(geom.coords))
    else:
        boundary_coords = list(line_buffer.boundary.coords)
    all_points = np.vstack([points, boundary_coords])

    vor = Voronoi(all_points)
    tree = cKDTree(points)
    polys = []
    wl_values = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = Polygon([vor.vertices[i] for i in region])
            center = polygon.centroid.coords[0]
            _, idx = tree.query(center)
            polys.append(polygon)
            wl_values.append(df.iloc[idx]['WL'])

    gdf_voronois = gpd.GeoDataFrame({'geometry': polys, 'WL': wl_values}, crs="epsg:3857")
    clipped_voronois = gpd.clip(gdf_voronois, line_buffer)
    valid_geometries = clipped_voronois.geometry.apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))
    filtered_voronois = clipped_voronois[valid_geometries]

    dem_ds = gdal.Open(dem_file_path)
    dem_crs = osr.SpatialReference(wkt=dem_ds.GetProjection())
    if dem_crs is None:
        logging.error("Cannot fetch DEM 좌표계 system. Exiting.")
        sys.exit(1)

    dem_band = dem_ds.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray()
    dem_transform = dem_ds.GetGeoTransform()
    dem_projection = dem_ds.GetProjection()
    result_data = np.full(dem_data.shape, -9999, dtype=dem_data.dtype)
    dem_ds = None

    filtered_voronois_5186 = filtered_voronois.to_crs(epsg=int(dem_crs.GetAuthorityCode(None)))

    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')
    mem_layer = mem_ds.CreateLayer('mem_layer', srs=dem_crs, geom_type=ogr.wkbPolygon)

    field_defn = ogr.FieldDefn('WL', ogr.OFTReal)
    mem_layer.CreateField(field_defn)

    for geom, wl in zip(filtered_voronois_5186.geometry, filtered_voronois_5186['WL']):
        feat = ogr.Feature(mem_layer.GetLayerDefn())
        feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
        feat.SetField('WL', wl)
        mem_layer.CreateFeature(feat)
        feat = None

    mask_ds = gdal.GetDriverByName('MEM').Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(dem_transform)
    mask_ds.SetProjection(dem_projection)
    gdal.RasterizeLayer(mask_ds, [1], mem_layer, options=["ATTRIBUTE=WL"])

    mask = mask_ds.GetRasterBand(1).ReadAsArray().astype(float)
    comparison_mask = (mask != 0) & (dem_data < mask)
    result_data[comparison_mask] = mask[comparison_mask] - dem_data[comparison_mask]
    mask_ds = None

    return result_data, dem_transform, dem_projection

def load_config(filepath):
    config = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split(':', 1)
                config[key.strip()] = value.strip()
    return config

def merge_tifs(tif_files, output_path):
    vrt_path = output_path.replace('.tif', '.vrt')
    gdal.BuildVRT(vrt_path, tif_files)
    gdal.Translate(output_path, vrt_path)
    os.remove(vrt_path)

def save_floodmap(result_data, dem_transform, dem_projection, output_path):
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_path, result_data.shape[1], result_data.shape[0], 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(dem_transform)
    dst_ds.SetProjection(dem_projection)
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(-9999)
    dst_band.WriteArray(result_data)
    dst_ds.FlushCache()
    dst_ds = None

    flood_area_cells = np.sum(result_data != -9999)
    pixel_width = abs(dem_transform[1])
    pixel_height = abs(dem_transform[5])
    flood_area = round((flood_area_cells * pixel_width * pixel_height) / 1_000_000, 2)
    max_depth = round(np.max(result_data[result_data != -9999]), 3)

    return flood_area_cells, flood_area, max_depth

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler(sys.stdout)])

def save_area_info(output_path, cell_size, timestamp, flood_area, max_depth):
    with open(output_path, 'a') as file:
        file.write(f"Timestamp: {timestamp}, Cell Size: {cell_size}, Flood Area: {flood_area} km², Max Depth: {max_depth} m\n")

# 고정된 컬러맵 생성 함수
def create_fixed_color_relief_file(output_path):
    elevation_steps = np.linspace(0, 12, num=6)
    colors = [
        (0, 0, 255),   # Blue
        (0, 255, 255), # Cyan
        (0, 255, 0),   # Green
        (255, 255, 0), # Yellow
        (255, 0, 0),   # Red
        (255, 0, 255)  # Magenta
    ]

    with open(output_path, 'w') as file:
        file.write('-9999 0 0 0 0\n')  # No data value를 투명하게 설정
        for step, color in zip(elevation_steps, colors):
            file.write(f"{step} {color[0]} {color[1]} {color[2]}\n")
        file.write('12 255 255 255 255\n')  # 12보다 큰 값은 흰색으로 설정

def create_legend(legend_output_path):
    # 고정된 고도와 색상 정보를 사용합니다.
    elevation_steps = np.linspace(0, 12, num=7)
    colors = [
        (0, 0, 255),   # Blue
        (0, 255, 255), # Cyan
        (0, 255, 0),   # Green
        (255, 255, 0), # Yellow
        (255, 0, 0),   # Red
        (255, 0, 255)  # Magenta
    ]

    # 색상 값을 [0, 1] 범위로 정규화합니다.
    colors = np.array(colors) / 255.0

    # 색상 목록으로 컬러맵을 생성합니다.
    cmap = mcolors.ListedColormap(colors)
    # 고도 범위에 맞는 색상 경계를 설정합니다.
    norm = mcolors.BoundaryNorm(elevation_steps, cmap.N)

    # 새로운 그림과 축을 생성합니다.
    fig = plt.figure(figsize=(2, 8))
    ax = fig.add_axes([0.05, 0.05, 0.15, 0.9])  # 축 위치 및 크기 조정

    # ScalarMappable 객체를 생성하여 컬러맵과 정규화를 적용합니다.
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 색상바를 생성하여 축에 추가합니다.
    cb = fig.colorbar(sm, cax=ax, orientation='vertical', spacing='proportional', ticks=elevation_steps, format='%1.2f')
    # 색상바의 레이블을 설정합니다.
    cb.set_label('Depth (m)', fontsize=20, color='black')  # 레이블 글씨 크기와 색상 변경
    # 색상바 눈금의 색상을 흰색으로 설정합니다.
    cb.ax.yaxis.set_tick_params(labelsize=20, color='black')  # 눈금 글씨 크기와 색상 변경
    # 색상바의 눈금 레이블 색상을 흰색으로 설정합니다.
    plt.setp(plt.getp(cb.ax, 'yticklabels'), fontsize=20, color='black')  # 눈금 레이블 글씨 크기와 색상 변경

    # 그림의 배경색을 투명하게 설정합니다.
    fig.patch.set_alpha(0.0)  # 투명하게 설정
    # 축의 배경색을 투명하게 설정합니다.
    ax.set_facecolor('none')  # 투명하게 설정

    # 그림을 파일로 저장합니다.
    plt.savefig(legend_output_path, bbox_inches='tight', pad_inches=0.1, transparent=True)
    # 그림을 닫아 자원을 해제합니다.
    plt.close()

def process_files(river, tributaries, start, end, dem_file_path, buffer_distances, progress_var, progress_bar):
    start_dt = datetime.strptime(start, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end, "%Y%m%d%H%M")
    current_dt = start_dt

    setup_environment()

    script_dir = os.path.dirname(__file__)
    log_file = os.path.join(script_dir, 'process.log')
    setup_logging(log_file)

    result_folder = os.path.join(script_dir, 'FIM_result')
    os.makedirs(result_folder, exist_ok=True)

    sample_dem_ds = gdal.Open(dem_file_path)
    sample_dem_transform = sample_dem_ds.GetGeoTransform()
    pixel_width = abs(sample_dem_transform[1])
    pixel_height = abs(sample_dem_transform[5])
    cell_size = f"Width = {pixel_width}, Height = {pixel_height}"
    logging.info(f"DEM Cell Size: {cell_size}")

    area_info_file = os.path.join(script_dir, 'area_info.txt')
    with open(area_info_file, 'w') as file:
        file.write(f"Cell Size: {cell_size}\n")

    sample_dem_ds = None

    total_hours = int((end_dt - start_dt).total_seconds() // 3600)
    total_steps = total_hours
    if total_steps == 0:
        total_steps = 1  # To avoid division by zero

    current_step = 0

    legend_output_path = os.path.join(result_folder, 'color_legend.png')
    color_relief_file = os.path.join(result_folder, 'color_relief.txt')
    
    # 고정된 color_relief 파일 생성
    create_fixed_color_relief_file(color_relief_file)
    create_legend(legend_output_path)
    
    while current_dt <= end_dt:
        timestamp = current_dt.strftime("%Y%m%d%H%M")
        floodmap_results = []
        dem_transform = None
        dem_projection = None

        with ThreadPoolExecutor() as executor:
            futures = []
            for idx in range(tributaries):
                river_branch_name = f"{river}_{idx+1}"
                input_path = os.path.join(script_dir, 'FIM_input_data', river_branch_name, f"{river_branch_name}_{timestamp}.csv")

                if os.path.exists(input_path):
                    buffer_size = buffer_distances[idx]
                    futures.append(executor.submit(process_file, input_path, buffer_size, dem_file_path))
                else:
                    logging.warning(f"Cannot find file: {input_path}")

            for future in futures:
                try:
                    result_data, transform, projection = future.result()
                    floodmap_results.append(result_data)
                    if dem_transform is None:
                        dem_transform = transform
                    if dem_projection is None:
                        dem_projection = projection
                except Exception as e:
                    logging.error(f"Error processing file: {e}")

        if floodmap_results:
            merged_data = np.max(np.array(floodmap_results), axis=0)
            temp_output_path = os.path.join(result_folder, f"Temp_Floodmap_{timestamp}.tif")
            merged_output_path = os.path.join(result_folder, f"Floodmap_{river}_{timestamp}.tif")
            flood_area_cells, flood_area, max_depth = save_floodmap(merged_data, dem_transform, dem_projection, temp_output_path)

            if flood_area_cells is not None:
                logging.info(f"Merged flood map for timestamp {timestamp}: {temp_output_path}")
                logging.info(f"Flood area (valid cell count) for timestamp {timestamp}: {flood_area_cells}")
                logging.info(f"Flood area (square kilometers) for timestamp {timestamp}: {flood_area} km²")
                logging.info(f"Max flood depth for timestamp {timestamp}: {max_depth} m")
                save_area_info(area_info_file, cell_size, timestamp, flood_area, max_depth)

                subprocess.run([
                    'gdaldem', 'color-relief',
                    temp_output_path,
                    color_relief_file,
                    merged_output_path,
                    '-alpha'
                ], check=True)
                logging.info(f"Colorful flood map for timestamp {timestamp}: {merged_output_path}")

                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        # 진행률 업데이트
        current_step += 1
        update_progress(progress_bar, progress_var, current_step, total_steps)

        current_dt += timedelta(hours=1)

    messagebox.showinfo("Success", "홍수피해면적_산출완료")

# 통합된 GUI 설정

root = tk.Tk()
root.title("파일 선택 및 버퍼 크기 설정")

river_name = tk.StringVar()
tributary_count = tk.IntVar()
start_time = tk.StringVar()
end_time = tk.StringVar()
buffer_distances = []
default_buffer_distance = tk.DoubleVar()

show_tributary_fields = tk.BooleanVar(value=False)

def toggle_tributary_fields():
    global show_tributary_fields
    show_tributary_fields.set(not show_tributary_fields.get())
    if show_tributary_fields.get():
        add_tributary_fields()
    else:
        for widget in tributary_frame.winfo_children():
            widget.destroy()

def add_tributary_fields():
    for widget in tributary_frame.winfo_children():
        widget.destroy()

    buffer_distances.clear()

    for i in range(tributary_count.get()):
        tk.Label(tributary_frame, text=f"지류 {i+1} 이름:").grid(row=i, column=0)
        tk.Entry(tributary_frame, textvariable=tk.StringVar(value=f"{river_name.get()}{i+1}")).grid(row=i, column=1)
        tk.Label(tributary_frame, text=f"버퍼 크기:").grid(row=i, column=2)
        buffer_distance = tk.DoubleVar()
        buffer_distances.append(buffer_distance)
        tk.Entry(tributary_frame, textvariable=buffer_distance).grid(row=i, column=3)

def select_dem_file():
    global dem_file_path
    file_path = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")])
    if file_path:
        dem_file_path = file_path
        dem_path_label.config(text=os.path.basename(file_path))
    else:
        dem_path_label.config(text="No file selected.")

def start_create_input_files():
    threading.Thread(target=create_input_files, args=(main_progress_var, main_progress_bar)).start()

def start_process_files():
    river = river_name.get()
    tributaries = tributary_count.get()
    start = start_time.get()
    end = end_time.get()

    if not buffer_distances or not show_tributary_fields.get():
        buffer_distances_list = [default_buffer_distance.get()] * tributaries
    else:
        buffer_distances_list = [dist.get() for dist in buffer_distances]

    if not river or tributaries <= 0 or not start or not end or any(dist <= 0.0 for dist in buffer_distances_list):
        messagebox.showerror("Error", "All fields must be correctly filled.")
        return

    threading.Thread(target=process_files, args=(river, tributaries, start, end, dem_file_path, buffer_distances_list, main_progress_var, main_progress_bar)).start()

# Create input files button
input_files_button = tk.Button(root, text="Create Input Files", command=start_create_input_files)
input_files_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

# GUI Components
tk.Label(root, text="River Name:").grid(row=1, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=river_name).grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Number of Tributaries:").grid(row=2, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=tributary_count).grid(row=2, column=1, padx=5, pady=5)

add_fields_button = tk.Button(root, text="Toggle Tributary Fields", command=toggle_tributary_fields)
add_fields_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

tributary_frame = tk.Frame(root)
tributary_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

tk.Label(root, text="Default Buffer Distance:").grid(row=5, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=default_buffer_distance).grid(row=5, column=1, padx=5, pady=5)

tk.Label(root, text="Start Time (YYYYMMDDhhmm):").grid(row=6, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=start_time).grid(row=6, column=1, padx=5, pady=5)

tk.Label(root, text="End Time (YYYYMMDDhhmm):").grid(row=7, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=end_time).grid(row=7, column=1, padx=5, pady=5)

dem_button = tk.Button(root, text="Select DEM File", command=select_dem_file)
dem_button.grid(row=8, column=0, columnspan=1, padx=5, pady=5)

dem_path_label = tk.Label(root, text="No DEM file selected.")
dem_path_label.grid(row=8, column=1, columnspan=2, padx=5, pady=5)

process_button = tk.Button(root, text="Start Processing", command=start_process_files)
process_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

# 진행률 표시줄 추가
main_progress_var = StringVar()
main_progress_bar = Progressbar(root, length=300, mode='determinate')
main_progress_bar.grid(row=10, column=0, columnspan=2, pady=10)
main_progress_percentage_label = Label(root, textvariable=main_progress_var)
main_progress_percentage_label.grid(row=11, column=0, columnspan=2, pady=5)

# 터미널 출력을 위한 텍스트 위젯 추가
terminal_output = Text(root, height=15, width=80, bg='black', fg='white')
terminal_output.grid(row=12, column=0, columnspan=2, padx=5, pady=5)
scrollbar = Scrollbar(root, command=terminal_output.yview, orient=VERTICAL)
scrollbar.grid(row=12, column=2, sticky='ns')
terminal_output.configure(yscrollcommand=scrollbar.set)

# 터미널 출력을 리다이렉트
redirect_text = RedirectText(terminal_output)
sys.stdout = redirect_text
sys.stderr = redirect_text

root.mainloop()
