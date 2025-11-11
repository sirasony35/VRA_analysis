# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import geopandas as gpd

# --- 1. 사용자 설정 부분 ---
QGIS_INSTALL_PATH = 'C:/Program Files/QGIS 3.40.11'  # 버전 업데이트 반영
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
    """(1, 2단계용) 5개의 고유 DN 값을 찾아 범주형(Categorized) 렌더러 생성 (요구사항 반영)"""
    from qgis.core import QgsCategorizedSymbolRenderer, QgsSymbol, QgsRendererCategory
    from PyQt5.QtGui import QColor

    # DN 값 0을 제외하고 내림차순 정렬
    unique_values = layer.uniqueValues(layer.fields().indexFromName(field_name))
    dn_values = sorted([val for val in unique_values if val != 0], reverse=True)

    if len(dn_values) > 5:
        print(f"   [경고] DN 값이 5개 이상입니다. 상위 5개 값만 사용합니다: {dn_values[:5]}")
        dn_values = dn_values[:5]
    elif len(dn_values) < 5:
        print(f"   [경고] DN 값이 5개 미만입니다 (총 {len(dn_values)}개).")

    # 요구사항 색상: 1(빨강) ~ 5(파랑)
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']  # 높(빨) ~ 낮(파)
    categories = []

    for i, value in enumerate(dn_values):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        if i < len(colors):
            symbol.setColor(QColor(colors[i]))
        else:
            # 5개가 넘어가거나 모자랄 경우를 대비한 기본값
            symbol.setColor(QColor("#808080"))
        category = QgsRendererCategory(value, symbol, str(value))
        categories.append(category)

    renderer = QgsCategorizedSymbolRenderer(field_name, categories)
    return renderer


def set_labeling(layer, field_name, format_type='decimal'):
    """레이어에 라벨을 설정하는 함수 (정수 또는 소수점 3자리) (요구사항 반영)"""
    from qgis.core import QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat

    layer_settings = QgsPalLayerSettings()
    text_format = QgsTextFormat()
    layer_settings.setFormat(text_format)

    if format_type == 'integer':
        # 정수 표시 (DN 값)
        expression = f"format_number( \"{field_name}\", 0)"
    else:
        # 소수점 3자리 표시 (before_mean, after_mean)
        expression = f"format_number( \"{field_name}\", 3)"

    layer_settings.fieldName = expression
    layer_settings.isExpression = True

    labeling = QgsVectorLayerSimpleLabeling(layer_settings)
    layer.setLabelsEnabled(True)
    layer.setLabeling(labeling)


def export_styled_layer_to_image(layer, output_path, scale=None):
    """스타일이 적용된 벡터 레이어를 직접 렌더링하여 PNG로 저장합니다. (요구사항 반영)"""
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob
    from PyQt5.QtCore import QSize, QEventLoop
    from PyQt5.QtGui import QColor

    map_settings = QgsMapSettings()
    map_settings.setLayers([layer])
    map_settings.setBackgroundColor(QColor(255, 255, 255, 0))  # 배경 투명
    extent = layer.extent()

    if scale:
        # [3단계] 요구사항: 축척 1:150 적용
        map_settings.setMapScale(scale)
        map_settings.setCenter(extent.center())
        # A4 가로 (297x210mm) 기준 150DPI 크기 계산 (참고용 크기)
        width_px = int(297 / 25.4 * 150)
        height_px = int(210 / 25.4 * 150)
        map_settings.setOutputSize(QSize(width_px, height_px))
    else:
        # [1, 2단계] 축척 미지정: 레이어 영역에 맞춤
        map_settings.setExtent(extent)
        width_px = 2000  # 기본 너비
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

    # QgsEqualIntervalClassification Import 완전히 제거 (ImportError 방지)
    from qgis.core import (QgsApplication, QgsVectorLayer, QgsRasterLayer,
                           QgsProject, QgsSymbol, QgsGraduatedSymbolRenderer,
                           QgsStyle)
    from PyQt5.QtCore import QEventLoop, QSize
    from PyQt5.QtGui import QColor

    from qgis.analysis import QgsNativeAlgorithms  # QgsEqualIntervalClassification Import 제거
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

    # [1-1] VRA 파일 불러오기
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

        # [1-2] 벡터 변환 (DN 필드)
        processing.run("gdal:polygonize", {
            'INPUT': vra_rx_file, 'BAND': 1, 'FIELD': 'DN',
            'EIGHT_CONNECTEDNESS': False, 'OUTPUT': vector_path
        })

        vra_vector_layer = QgsVectorLayer(vector_path, f"{field_code}_vra_vector", "ogr")
        if not vra_vector_layer.isValid():
            print(f"   [오류] 벡터 변환에 실패했습니다: {vector_path}")
            continue

        # [1-3] DN=0 필드 삭제
        vra_vector_layer.startEditing()
        vra_vector_layer.deleteFeatures([f.id() for f in vra_vector_layer.getFeatures() if f['DN'] == 0])
        vra_vector_layer.commitChanges()

        # [1-4] 심볼 변경 (Categorized)
        vra_vector_layer.setRenderer(get_categorized_renderer(vra_vector_layer, "DN"))
        # [1-5] 라벨 표시 (DN, 정수)
        set_labeling(vra_vector_layer, "DN", format_type='integer')

        # [1-6] 이미지로 저장
        img_path_1 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA.png")
        export_styled_layer_to_image(vra_vector_layer, img_path_1)
        print(f"   [성공] 1단계 이미지 저장: {img_path_1}")

        # === 2단계: 'Before' TIF 구역 통계 ===
        print("   [2단계] 'Before' TIF 구역 통계 및 스타일링...")
        # [2-1] Before TIF 파일 불러오기
        before_raster_path = find_data_file(field_code, BEFORE_FOLDER)
        if not before_raster_path:
            continue

        before_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_before.gpkg")

        # [2-2] 구역 통계 (접두어 'before_')
        # [★개선사항] STATISTICS: Mean(2), Median(3), Min(5), Max(6), StdDev(4)
        processing.run("native:zonalstatisticsfb", {
            'INPUT_RASTER': before_raster_path, 'INPUT': vector_path, 'BAND': 1,
            'COLUMN_PREFIX': 'before_', 'STATISTICS': [2, 3, 5, 6, 4],
            'OUTPUT': before_stats_path
        })

        # [2-3] 레이어 이름 설정
        vra_before_layer = QgsVectorLayer(before_stats_path, f"{field_code}_vra_before", "ogr")
        if not vra_before_layer.isValid():
            print(f"   [오류] 'Before' 구역 통계에 실패했습니다: {before_stats_path}")
            continue

        # [2-4] 심볼 설정 (1단계와 동일)
        vra_before_layer.setRenderer(get_categorized_renderer(vra_before_layer, "DN"))
        # [2-5] 라벨 표시 (before_mean, 소수점 3자리)
        set_labeling(vra_before_layer, "before_mean", format_type='decimal')

        # [2-6] 이미지로 저장
        img_path_2 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_before.png")
        export_styled_layer_to_image(vra_before_layer, img_path_2)
        print(f"   [성공] 2단계 이미지 저장: {img_path_2}")

        # === 3단계: 'After' TIF 구역 통계 ===
        print("   [3단계] 'After' TIF 구역 통계 실행...")
        # [3-1] After TIF 파일 불러오기
        after_raster_path = find_data_file(field_code, AFTER_FOLDER)
        if not after_raster_path:
            continue

        # [3-3] 산출물 이름 설정
        after_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_after.gpkg")

        # [3-2] 구역 통계 (입력: vra_before, 접두어: 'after_')
        # [★개선사항] STATISTICS: Mean(2), Median(3), Min(5), Max(6), StdDev(4)
        processing.run("native:zonalstatisticsfb", {
            'INPUT_RASTER': after_raster_path, 'INPUT': before_stats_path, 'BAND': 1,
            'COLUMN_PREFIX': 'after_', 'STATISTICS': [2, 3, 5, 6, 4],
            'OUTPUT': after_stats_path
        })

        vra_after_layer = QgsVectorLayer(after_stats_path, f"{field_code}_vra_after", "ogr")
        if not vra_after_layer.isValid():
            print(f"   [오류] 'After' 구역 통계에 실패했습니다: {after_stats_path}")
            continue

        # === 4단계: CSV 데이터 집계 (요구사항 순서: 이미지 생성 전) ===
        print("   [4단계] CSV 데이터 집계...")
        gdf = gpd.read_file(after_stats_path)

        # [4-1] 기본 통계 (표준편차)
        before_mean_std = gdf['before_mean'].std()
        after_mean_std = gdf['after_mean'].std()
        cv_data_list.append({
            '필지코드': field_code,
            'before_mean_cv': before_mean_std,  # 요구사항 컬럼명 'before_mean_cv'
            'after_mean_cv': after_mean_std  # 요구사항 컬럼명 'before_mean_cv'
        })

        # [4-2] 영역별 데이터 (그룹별)
        # DN 그룹과 색상 이름 매핑 (1단계와 동일한 로직)
        unique_dns_sorted = sorted([val for val in gdf['DN'].unique() if val != 0], reverse=True)
        dn_color_names = ['빨강색', '주황색', '베이지색', '초록색', '파란색']
        dn_to_color_map = {}
        for i, dn_val in enumerate(unique_dns_sorted):
            if i < len(dn_color_names):
                dn_to_color_map[dn_val] = dn_color_names[i]
            else:
                dn_to_color_map[dn_val] = '기타'  # 5개 초과시

        grouped = gdf.groupby('DN')
        for dn_value, group in grouped:
            if dn_value == 0: continue  # DN=0은 이미 삭제했지만 안전장치

            before_mean_result = group['before_mean'].mean()
            after_mean_result = group['after_mean'].mean()

            vi_rate = 0
            if before_mean_result != 0 and before_mean_result is not None:
                # 증감율 계산
                vi_rate = (after_mean_result - before_mean_result) / before_mean_result * 100

            # color_group 컬럼 생성
            color_group_name = dn_to_color_map.get(dn_value, '알 수 없음')

            growth_data_list.append({
                '필지코드': field_code,
                'color_group': color_group_name,
                'before_mean_result': before_mean_result,
                'after_mean_result': after_mean_result,
                'vi_rate': vi_rate
            })
        print(f"   [성공] {field_code} 필지 통계 집계 완료.")

        # === 5단계 (구 3단계): 'After' 이미지 생성 ===
        print("   [5단계] 'After' 레이어 스타일링 및 이미지 저장...")

        symbol = QgsSymbol.defaultSymbol(vra_after_layer.geometryType())
        color_ramp = QgsStyle.defaultStyle().colorRamp('Spectral')  # [3-4] Spectral 램프

        # === ★★★ 최종 안정화 수정 (2025.11.11) - updateClasses(layer) 사용 ★★★

        # 1. 기본 생성자로 렌더러 객체 생성
        renderer = QgsGraduatedSymbolRenderer()

        # 2. 분류 기준으로 사용할 필드 설정
        renderer.setClassAttribute('after_mean')

        # 3. 분류 모드 설정 (Warning 발생 가능하지만, TypeError 우회를 위한 필수 설정)
        renderer.setMode(QgsGraduatedSymbolRenderer.EqualInterval)

        renderer.setSourceSymbol(symbol)
        renderer.setSourceColorRamp(color_ramp)

        # 4. 분류 실행 (레이어 객체만 인자로 전달)
        #    이것이 'not enough arguments'와 'TypeError'를 우회하는 마지막 시도입니다.
        renderer.updateClasses(vra_after_layer)

        # === 수정 완료 ===

        vra_after_layer.setRenderer(renderer)

        # [3-5] 라벨 설정 (after_mean, 소수점 3자리)
        set_labeling(vra_after_layer, "after_mean", format_type='decimal')

        # [3-6] 이미지 저장 (축척 1:150)
        img_path_3 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_after.png")
        export_styled_layer_to_image(vra_after_layer, img_path_3, scale=150)
        print(f"   [성공] 3단계(After) 이미지 저장: {img_path_3}")

        # 레이어 객체 삭제 (메모리 관리)
        del vra_vector_layer
        del vra_before_layer
        del vra_after_layer

    print("\n--- 모든 필지 처리 완료 ---")

    # === 6단계: 최종 CSV 파일 저장 (루프 밖) ===

    # [4-1] CSV 저장
    csv_path_1 = os.path.join(OUTPUT_CSV_FOLDER, '변량시비 전후 표준편차.csv')
    df1 = pd.DataFrame(cv_data_list)
    df1.to_csv(csv_path_1, index=False, encoding='utf-8-sig')
    print(f"[최종 성공] CV 통계 CSV 파일 저장 완료: {csv_path_1}")

    # [4-2] CSV 저장
    csv_path_2 = os.path.join(OUTPUT_CSV_FOLDER, '변량시비 전후 생육 변화량 확인.csv')
    df2 = pd.DataFrame(growth_data_list)
    df2.to_csv(csv_path_2, index=False, encoding='utf-8-sig')
    print(f"[최종 성공] 생육 변화량 CSV 파일 저장 완료: {csv_path_2}")

    qgs.exitQgis()
    print("\n--- 모든 작업이 완료되었습니다. ---")

    print(
        "\n만약 이 코드가 여전히 5단계에서 오류를 발생시킨다면, 이는 해당 QGIS 버전(3.40.11)의 파이썬 바인딩에 근본적인 버그가 있는 것으로 판단해야 합니다. 이 경우, QGIS GUI에서 직접 스타일을 저장하여 파일로 불러오는 방식(QML)을 고려해야 합니다.")


if __name__ == '__main__':
    main()