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

# --- (여기부터는 수정할 필요 없습니다) ---

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


# === ★★★ 최종 수정된 'get_graduated_quantile_renderer' 함수 ★★★ ===
def get_graduated_quantile_renderer(layer, field_name):
    """
    (1, 2단계용) 'DN' 필드를 기준으로 5단계(Quantile) 단계 구분 렌더러 생성
    요구사항: 높음(빨강) ~ 낮음(파랑)
    """
    from qgis.core import (QgsGraduatedSymbolRenderer, QgsSymbol, QgsRendererRange,
                           QgsClassificationQuantile)
    from PyQt5.QtGui import QColor

    # 1. 색상 정의 (낮음 -> 높음 순서)
    colors = [
        QColor("#2b83ba"),  # 파랑 (가장 낮음)
        QColor("#abdda4"),  # 초록
        QColor("#ffffbf"),  # 베이지
        QColor("#fdae61"),  # 주황
        QColor("#d7191c")  # 빨강 (가장 높음)
    ]

    # 2. 분류기(Quantile, 5단계) 생성
    classifier = QgsClassificationQuantile()

    # 3. DN=0인 피처를 제외하도록 필터 설정
    classifier.setFilterExpression('"DN" > 0')

    # 4. 'classes' 메서드를 호출하여 5개의 QgsRendererRange 리스트를 가져옵니다.
    #    (오류가 발생했던 'classify' 메서드 대신 'classes'를 사용)
    try:
        ranges = classifier.classes(layer, field_name, 5)
    except Exception as e:
        print(f"   [오류] Quantile 분류 중 오류 발생: {e} (데이터가 너무 적거나 고유 값이 없을 수 있습니다.)")
        ranges = []  # 빈 리스트로 계속 진행

    # 5. 렌더러 생성
    renderer = QgsGraduatedSymbolRenderer(field_name)

    # 6. 가져온 'ranges'에 사용자 정의 색상을 적용
    final_ranges = []

    # 분류된 범위가 5개보다 적을 수 있음 (고유 값 부족 등)
    if 0 < len(ranges) <= 5:
        for i, rng in enumerate(ranges):
            # i가 color 리스트의 인덱스를 넘어가지 않도록 보장
            color_index = min(i, len(colors) - 1)

            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(colors[color_index])
            rng.setSymbol(symbol)  # Apply the new symbol with the correct color
            final_ranges.append(rng)

    elif len(ranges) > 5:  # 5개 초과 (드문 경우)
        print(f"   [경고] {layer.name()}: 5개 이상의 범위({len(ranges)}개)가 반환되었습니다. 5개로 자릅니다.")
        for i in range(5):
            rng = ranges[i]
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(colors[i])
            rng.setSymbol(symbol)
            final_ranges.append(rng)

    else:  # len(ranges) == 0
        print(f"   [경고] {layer.name()}: DN=0 제외 후 5단계 분류에 실패했습니다. (범위 0개 반환)")
        # final_ranges는 빈 상태로 유지됨

    renderer.setRanges(final_ranges)
    return renderer


# === ★★★ 수정 완료 ★★★ ===


def set_labeling(layer, field_name, format_type='decimal'):
    """레이어에 라벨을 설정하는 함수 (정수 또는 소수점 3자리) - (원본 유지)"""
    from qgis.core import QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat

    layer_settings = QgsPalLayerSettings()
    text_format = QgsTextFormat()
    layer_settings.setFormat(text_format)

    if format_type == 'integer':
        # 1단계 요구사항: 숫자 정수로 표시
        expression = f"format_number( \"{field_name}\", 0)"
    else:
        # 2, 3단계 요구사항: 숫자 소수점 3자리 까지 표시
        expression = f"format_number( \"{field_name}\", 3)"

    layer_settings.fieldName = expression
    layer_settings.isExpression = True

    labeling = QgsVectorLayerSimpleLabeling(layer_settings)
    layer.setLabelsEnabled(True)
    layer.setLabeling(labeling)


def export_styled_layer_to_image(layer, output_path, scale=None):
    """스타일이 적용된 벡터 레이어를 직접 렌더링하여 PNG로 저장합니다. - (원본 유지)"""
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob
    from PyQt5.QtCore import QSize, QEventLoop
    from PyQt5.QtGui import QColor

    map_settings = QgsMapSettings()
    map_settings.setLayers([layer])
    map_settings.setBackgroundColor(QColor(255, 255, 255, 0))  # 배경 투명
    extent = layer.extent()

    if scale:  # 3단계: 1:150 축척 적용
        map_settings.setMapScale(scale)
        map_settings.setCenter(extent.center())
        # QGIS 기본 DPI(96) 기준으로 렌더링 크기 계산 (약 A4 크기)
        width_px = int(297 / 25.4 * 96 * (150 / 96))
        height_px = int(210 / 25.4 * 96 * (150 / 96))
        map_settings.setOutputSize(QSize(width_px, height_px))
    else:  # 1, 2단계: 범위에 맞춤
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

    # ★★★ QgsClassificationEqualInterval/Quantile를 import 목록에 추가 ★★★
    from qgis.core import (QgsApplication, QgsVectorLayer, QgsRasterLayer,
                           QgsProject, QgsSymbol, QgsGraduatedSymbolRenderer,
                           QgsStyle, QgsClassificationEqualInterval, QgsClassificationQuantile)
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

        # === ★★★ 수정된 부분 2: 단계 구분 렌더러 적용 ★★★ ===
        vra_vector_layer.setRenderer(get_graduated_quantile_renderer(vra_vector_layer, "DN"))
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

        # === ★★★ 수정된 부분 3: 구역 통계 항목 변경 ★★★ ===
        # Mean(2), Median(3), Min(5), Max(6), StdDev(4)
        processing.run("native:zonalstatisticsfb", {
            'INPUT_RASTER': before_raster_path, 'INPUT': vector_path, 'BAND': 1,
            'COLUMN_PREFIX': 'before_', 'STATISTICS': [2, 3, 5, 6, 4], 'OUTPUT': before_stats_path
        })

        vra_before_layer = QgsVectorLayer(before_stats_path, f"{field_code}_vra_before", "ogr")
        if not vra_before_layer.isValid():
            print(f"   [오류] 'Before' 구역 통계에 실패했습니다: {before_stats_path}")
            continue

        # === ★★★ 수정된 부분 4: 단계 구분 렌더러 적용 ★★★ ===
        vra_before_layer.setRenderer(get_graduated_quantile_renderer(vra_before_layer, "DN"))
        set_labeling(vra_before_layer, "before_mean", format_type='decimal')

        img_path_2 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_before.png")
        export_styled_layer_to_image(vra_before_layer, img_path_2)
        print(f"   [성공] 2단계 이미지 저장: {img_path_2}")

        # === 3단계: 'After' TIF 구역 통계 ===
        print("   [3단계] 'After' TIF 구역 통계 및 스타일링...")
        after_raster_path = find_data_file(field_code, AFTER_FOLDER)
        if not after_raster_path:
            continue

        after_stats_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vra_after.gpkg")

        # === ★★★ 수정된 부분 5: 구역 통계 항목 변경 ★★★ ===
        # Mean(2), Median(3), Min(5), Max(6), StdDev(4)
        processing.run("native:zonalstatisticsfb", {
            'INPUT_RASTER': after_raster_path, 'INPUT': before_stats_path, 'BAND': 1,
            'COLUMN_PREFIX': 'after_', 'STATISTICS': [2, 3, 5, 6, 4], 'OUTPUT': after_stats_path
        })

        vra_after_layer = QgsVectorLayer(after_stats_path, f"{field_code}_vra_after", "ogr")
        if not vra_after_layer.isValid():
            print(f"   [오류] 'After' 구역 통계에 실패했습니다: {after_stats_path}")
            continue

        # 3단계 렌더러 (Equal Interval) - (원본 유지, 올바르게 작성되었습니다)
        symbol = QgsSymbol.defaultSymbol(vra_after_layer.geometryType())
        color_ramp = QgsStyle.defaultStyle().colorRamp('Spectral')
        renderer = QgsGraduatedSymbolRenderer('after_mean')
        renderer.setSourceSymbol(symbol)
        renderer.setSourceColorRamp(color_ramp)
        classification_method = QgsClassificationEqualInterval()
        renderer.updateClasses(vra_after_layer, classification_method, 5)

        vra_after_layer.setRenderer(renderer)
        set_labeling(vra_after_layer, "after_mean", format_type='decimal')

        img_path_3 = os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_after.png")
        export_styled_layer_to_image(vra_after_layer, img_path_3, scale=150)
        print(f"   [성공] 3단계 이미지 저장: {img_path_3}")

        # === 4단계: CSV 데이터 집계 ===
        print("   [4단계] CSV 데이터 집계...")
        gdf = gpd.read_file(after_stats_path)

        # DN=0인 데이터는 통계에서 제외
        gdf = gdf[gdf['DN'] > 0].copy()

        # 1. 기본 통계 데이터 생성 (원본 유지)
        before_mean_cv = gdf['before_mean'].std()
        after_mean_cv = gdf['after_mean'].std()
        cv_data_list.append({
            '필지코드': field_code,
            'before_mean_cv': before_mean_cv,
            'after_mean_cv': after_mean_cv
        })

        # === ★★★ 수정된 부분 6: Quantile 등급(색상 그룹) 기준으로 통계 집계 ★★★ ===

        # 2. 영역별 데이터 생성 (그룹 통계)

        # DN 값을 기준으로 5개 Quantile 그룹 생성
        # 'duplicates=drop'은 동일한 경계 값이 많을 때 등급 수를 줄여서 처리
        try:
            color_group_labels = ['Blue', 'Green', 'Beige', 'Orange', 'Red']  # 낮음 -> 높음
            gdf['color_group'] = pd.qcut(gdf['DN'], 5, labels=color_group_labels, duplicates='drop')
        except ValueError:
            # 피처 수가 5개 미만이거나 모든 값이 동일할 경우
            print(f"   [경고] {field_code}의 DN 값 분포가 고유하지 않아 5개 그룹으로 나눌 수 없습니다.")
            gdf['color_group'] = 'Group 1'  # 단일 그룹으로 처리

        grouped = gdf.groupby('color_group', observed=True)

        for group_name, group_data in grouped:
            before_mean_result = group_data['before_mean'].mean()
            after_mean_result = group_data['after_mean'].mean()
            vi_rate = 0
            if before_mean_result != 0 and pd.notna(before_mean_result):
                vi_rate = (after_mean_result - before_mean_result) / before_mean_result * 100

            growth_data_list.append({
                '필지코드': field_code,
                'color_group': group_name,  # 요구사항 컬럼명 'color_group'
                'before_mean_result': before_mean_result,
                'after_mean_result': after_mean_result,
                'vi_rate': vi_rate
            })
        print(f"   [성공] {field_code} 필지 통계 집계 완료.")

        del vra_vector_layer
        del vra_before_layer
        del vra_after_layer

    print("\n--- 모든 필지 처리 완료 ---")

    # CSV 파일 저장 (원본 유지)
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