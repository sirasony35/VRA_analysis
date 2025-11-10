# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import geopandas as gpd

# --- 1. 사용자 설정 부분 ---
QGIS_INSTALL_PATH = 'C:/Program Files/QGIS 3.40.10'
VRA_FOLDER = 'vra_data'
BEFORE_FOLDER = 'before_data'
AFTER_FOLDER = 'after_data'
OUTPUT_IMAGE_FOLDER = 'result_images'
OUTPUT_CSV_FOLDER = 'result_csv'
OUTPUT_TEMP_FOLDER = 'temp_layers'


# -------------------------

def setup_qgis_environment():
    """QGIS 환경을 설정하는 함수"""
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python'))
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python/plugins'))
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/Qt5/plugins')
    os.environ['QT_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/qtplugins')
    print("QGIS 환경 설정 완료.")


def find_data_file(field_code, data_folder):
    """필지코드로 파일을 검색하되, 여러 개일 경우 GNDVI 파일을 우선 반환"""
    search_path = os.path.join(data_folder, f"{field_code}*.tif")
    files = glob.glob(search_path)
    if not files:
        print(f"   [오류] '{data_folder}'에서 '{field_code}'로 시작하는 파일을 찾을 수 없습니다.")
        return None
    if len(files) == 1:
        return files[0]
    for f in files:
        if 'GNDVI' in os.path.basename(f).upper():
            return f
    print(f"   [경고] '{data_folder}'에 '{field_code}' 파일이 여러 개지만 GNDVI 파일을 찾지 못했습니다. 첫 번째 파일을 사용합니다: {files[0]}")
    return files[0]


def get_categorized_renderer(layer, field_name):
    """(1, 2단계용) 5개의 고유 DN 값을 찾아 범주형(Categorized) 렌더러 생성"""
    from qgis.core import QgsCategorizedSymbolRenderer, QgsSymbol, QgsRendererCategory
    from PyQt5.QtGui import QColor

    unique_values = layer.uniqueValues(layer.fields().indexFromName(field_name))
    dn_values = sorted([val for val in unique_values if val != 0], reverse=True)
    if len(dn_values) > 5:
        print(f"   [경고] DN 값이 5개 이상입니다. 상위 5개 값만 사용합니다: {dn_values[:5]}")
        dn_values = dn_values[:5]
    elif len(dn_values) < 5:
        print(f"   [경고] DN 값이 5개 미만입니다 (총 {len(dn_values)}개).")

    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']  # 높(빨) ~ 낮(파)
    categories = []

    for i, value in enumerate(dn_values):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        if i < len(colors):
            symbol.setColor(QColor(colors[i]))
        else:
            symbol.setColor(QColor("#808080"))
        category = QgsRendererCategory(value, symbol, str(value))
        categories.append(category)
    renderer = QgsCategorizedSymbolRenderer(field_name, categories)
    return renderer


def set_labeling(layer, field_name, format_type='decimal'):
    """레이어에 라벨을 설정하는 함수 (정수 또는 소수점 3자리)"""
    from qgis.core import QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat

    layer_settings = QgsPalLayerSettings()
    text_format = QgsTextFormat()
    layer_settings.setFormat(text_format)

    if format_type == 'integer':
        expression = f"format_number( \"{field_name}\", 0)"
    else:
        expression = f"format_number( \"{field_name}\", 3)"

    layer_settings.fieldName = expression
    layer_settings.isExpression = True

    labeling = QgsVectorLayerSimpleLabeling(layer_settings)
    layer.setLabelsEnabled(True)
    layer.setLabeling(labeling)


def export_styled_layer_to_image(layer, output_path, scale=None):
    """스타일이 적용된 벡터 레이어를 직접 렌더링하여 PNG로 저장합니다. (백색 이미지 문제 해결)"""
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob
    from PyQt5.QtCore import QSize, QEventLoop
    from PyQt5.QtGui import QColor

    map_settings = QgsMapSettings()
    map_settings.setLayers([layer])
    map_settings.setBackgroundColor(QColor(255, 255, 255, 0))
    extent = layer.extent()

    if scale:
        map_settings.setMapScale(scale)
        map_settings.setCenter(extent.center())
        width_px = int(297 / 25.4 * 150)
        height_px = int(210 / 25.4 * 150)
        map_settings.setOutputSize(QSize(width_px, height_px))
    else:
        map_settings.setExtent(extent)
        width_px = 2000
        height_px = int(width_px * extent.height() / extent.width())
        map_settings.setOutputSize(QSize(width_px, height_px))

    job = QgsMapRendererParallelJob(map_settings)
    loop = QEventLoop()
    job.finished.connect(loop.quit)
    job.start()
    loop.exec_()

    image = job.renderedImage()
    image.save(output_path, "png")


def main():
    """메인 실행 함수"""
    print("QGIS 자동화 스크립트 실행 시작...")
    setup_qgis_environment()

    from qgis.core import (QgsApplication, QgsVectorLayer, QgsRasterLayer,
                           QgsProject, QgsSymbol, QgsGraduatedSymbolRenderer,
                           QgsStyle)
    from PyQt5.QtCore import QEventLoop, QSize
    from PyQt5.QtGui import QColor
    from qgis.analysis import QgsNativeAlgorithms
    import processing

    qgs = QgsApplication([], False)
    qgs.initQgis()

    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python/plugins'))
    from processing.core.Processing import Processing
    Processing.initialize()
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    project = QgsProject.instance()

    for folder in [OUTPUT_IMAGE_FOLDER, OUTPUT_CSV_FOLDER, OUTPUT_TEMP_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    cv_data_list = []
    growth_data_list = []

    vra_rx_files = glob.glob(os.path.join(VRA_FOLDER, '*_VRA_Rx.tif'))
    if not vra_rx_files:
        print(f"[오류] '{VRA_FOLDER}'에 '*_VRA_Rx.tif' 파일이 없습니다.")
        qgs.exitQgis()
        return

    print(f"\n총 {len(vra_rx_files)}개의 필지를 처리합니다.")

    for vra_rx_file in vra_rx_files:
        filename = os.path.basename(vra_rx_file)
        field_code = filename.split('_')[0]
        print(f"\n--- [{field_code}] 필지 처리 시작 ---")

        # === 1단계: VRA 처방맵 가공 ===
        print("   [1단계] VRA 래스터 벡터화 및 스타일링...")
        vector_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_vector.gpkg")
        processing.run("gdal:polygonize", {
            'INPUT': vra_rx_file, 'BAND': 1, 'FIELD': 'DN',
            'EIGHT_CONNECTEDNESS': False, 'OUTPUT': vector_path
        })

        vra_vector_layer = QgsVectorLayer(vector_path, f"{field_code}_vra_vector", "ogr")
        if not vra_vector_layer.isValid():
            print(f"   [오류] 벡터 변환에 실패했습니다: {vector_path}")
            continue

        vra_vector_layer.startEditing()
        vra_vector_layer.deleteFeatures([f.id() for f in vra_vector_layer.getFeatures() if f['DN'] == 0])
        vra_vector_layer.commitChanges()

        vra_vector_layer.setRenderer(get_categorized_renderer(vra_vector_layer, "DN"))
        set_labeling(vra_vector_layer, "DN", format_type='integer')

        img_path_1 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA.png")
        export_styled_layer_to_image(vra_vector_layer, img_path_1)
        print(f"   [성공] 1단계 이미지 저장: {img_path_1}")

        # === 2단계: 'Before' TIF 구역 통계 ===
        print("   [2단계] 'Before' TIF 구역 통계 및 스타일링...")
        before_raster_path = find_data_file(field_code, BEFORE_FOLDER)
        if not before_raster_path:
            continue

        before_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_before.gpkg")
        processing.run("native:zonalstatisticsfb", {
            'INPUT_RASTER': before_raster_path, 'INPUT': vector_path, 'BAND': 1,
            'COLUMN_PREFIX': 'before_', 'STATISTICS': [0, 1, 2, 3, 4], 'OUTPUT': before_stats_path
        })

        vra_before_layer = QgsVectorLayer(before_stats_path, f"{field_code}_vra_before", "ogr")
        if not vra_before_layer.isValid():
            print(f"   [오류] 'Before' 구역 통계에 실패했습니다: {before_stats_path}")
            continue

        vra_before_layer.setRenderer(get_categorized_renderer(vra_before_layer, "DN"))
        set_labeling(vra_before_layer, "before_mean", format_type='decimal')

        img_path_2 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_before.png")
        export_styled_layer_to_image(vra_before_layer, img_path_2)
        print(f"   [성공] 2단계 이미지 저장: {img_path_2}")

        # === 3단계: 'After' TIF 구역 통계 ===
        print("   [3단계] 'After' TIF 구역 통계 실행...")
        after_raster_path = find_data_file(field_code, AFTER_FOLDER)
        if not after_raster_path:
            continue

        after_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_after.gpkg")
        processing.run("native:zonalstatisticsfb", {
            'INPUT_RASTER': after_raster_path, 'INPUT': before_stats_path, 'BAND': 1,
            'COLUMN_PREFIX': 'after_', 'STATISTICS': [0, 1, 2, 3, 4], 'OUTPUT': after_stats_path
        })

        vra_after_layer = QgsVectorLayer(after_stats_path, f"{field_code}_vra_after", "ogr")
        if not vra_after_layer.isValid():
            print(f"   [오류] 'After' 구역 통계에 실패했습니다: {after_stats_path}")
            continue

        # === 4단계: CSV 데이터 집계 (이미지 생성 전으로 순서 변경) ===
        print("   [4단계] CSV 데이터 집계...")
        gdf = gpd.read_file(after_stats_path)

        before_mean_cv = gdf['before_mean'].std()
        after_mean_cv = gdf['after_mean'].std()
        cv_data_list.append({
            '필지코드': field_code,
            'before_mean_cv': before_mean_cv,
            'after_mean_cv': after_mean_cv
        })

        # 4-2. DN 그룹과 색상 이름 매핑
        unique_dns_sorted = sorted([val for val in gdf['DN'].unique() if val != 0], reverse=True)
        dn_color_names = ['빨강색', '주황색', '베이지색', '초록색', '파란색']
        dn_to_color_map = {}
        for i, dn_val in enumerate(unique_dns_sorted):
            if i < len(dn_color_names):
                dn_to_color_map[dn_val] = dn_color_names[i]
            else:
                dn_to_color_map[dn_val] = '기타'

        grouped = gdf.groupby('DN')
        for dn_value, group in grouped:
            before_mean_result = group['before_mean'].mean()
            after_mean_result = group['after_mean'].mean()
            vi_rate = 0
            if before_mean_result != 0:
                vi_rate = (after_mean_result - before_mean_result) / before_mean_result * 100

            color_group_name = dn_to_color_map.get(dn_value, '알 수 없음')

            growth_data_list.append({
                '필지코드': field_code,
                'color_group': color_group_name,  # 요구사항 'color_group' 컬럼
                'before_mean_result': before_mean_result,
                'after_mean_result': after_mean_result,
                'vi_rate': vi_rate
            })
        print(f"   [성공] {field_code} 필지 통계 집계 완료.")

        # === 5단계 (구 3단계): 'After' 이미지 생성 (EqualInterval 적용) ===
        print("   [5단계] 'After' 레이어 스타일링 및 이미지 저장...")
        symbol = QgsSymbol.defaultSymbol(vra_after_layer.geometryType())
        color_ramp = QgsStyle.defaultStyle().colorRamp('Spectral')

        # === ★★★ 수정된 부분: 'vra_after_layer' 인자 제거 ★★★
        renderer = QgsGraduatedSymbolRenderer.create(
            'after_mean', 5,  # 첫 번째 인자로 필드명 전달
            QgsGraduatedSymbolRenderer.EqualInterval, symbol, color_ramp
        )
        vra_after_layer.setRenderer(renderer)
        set_labeling(vra_after_layer, "after_mean", format_type='decimal')

        img_path_3 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_after.png")
        export_styled_layer_to_image(vra_after_layer, img_path_3, scale=150)
        print(f"   [성공] 3단계(After) 이미지 저장: {img_path_3}")

        # 레이어 객체 삭제 (메모리 관리)
        del vra_vector_layer
        del vra_before_layer
        del vra_after_layer

    print("\n--- 모든 필지 처리 완료 ---")

    # === 6단계: 최종 CSV 파일 저장 (루프 밖으로 이동) ===
    csv_path_1 = os.path.join(OUTPUT_CSV_FOLDER, '변량시비 전후 표준편차.csv')
    df1 = pd.DataFrame(cv_data_list)
    df1.to_csv(csv_path_1, index=False, encoding='utf-8-sig')
    print(f"[최종 성공] CV 통계 CSV 파일 저장 완료: {csv_path_1}")

    csv_path_2 = os.path.join(OUTPUT_CSV_FOLDER, '변량시비 전후 생육 변화량 확인.csv')
    df2 = pd.DataFrame(growth_data_list)
    df2.to_csv(csv_path_2, index=False, encoding='utf-8-sig')
    print(f"[최종 성공] 생육 변화량 CSV 파일 저장 완료: {csv_path_2}")

    qgs.exitQgis()
    print("\n--- 모든 작업이 완료되었습니다. ---")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()