# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import geopandas as gpd

# --- [사용자 설정] -------------------------------------------------
QGIS_INSTALL_PATH = 'C:/Program Files/QGIS 3.40.10'
VRA_FOLDER = 'vra_data'
BEFORE_FOLDER = 'before_data'
AFTER_FOLDER = 'after_data'
OUTPUT_IMAGE_FOLDER = 'result_images'
OUTPUT_CSV_FOLDER = 'result_csv'
OUTPUT_TEMP_FOLDER = 'temp_layers'
TARGET_CRS = 'EPSG:4326'   # 좌표계 통일 옵션
# ------------------------------------------------------------------

def setup_qgis_environment():
    """QGIS Standalone 환경 설정"""
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis/python'))
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis/python/plugins'))
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/Qt5/plugins')
    os.environ['QT_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/qgis/qtplugins')
    print("QGIS 환경 설정 완료.")


def find_data_file(field_code, data_folder):
    """필지코드로 TIF 파일 찾기 (GNDVI 우선)"""
    search_path = os.path.join(data_folder, f"{field_code}*.tif")
    files = glob.glob(search_path)
    if not files:
        print(f"   [오류] '{data_folder}'에서 '{field_code}' 파일을 찾을 수 없습니다.")
        return None
    for f in files:
        if 'GNDVI' in os.path.basename(f).upper():
            return f
    return files[0]


def get_categorized_renderer(layer, field_name):
    """DN 컬럼 기반 5단계 색상 렌더러 생성"""
    from qgis.core import QgsCategorizedSymbolRenderer, QgsSymbol, QgsRendererCategory
    from PyQt5.QtGui import QColor

    unique_values = sorted(list(layer.uniqueValues(layer.fields().indexFromName(field_name))), reverse=True)
    dn_values = [v for v in unique_values if v != 0][:5]
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
    categories = []
    for i, val in enumerate(dn_values):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QColor(colors[i] if i < len(colors) else "#808080"))
        categories.append(QgsRendererCategory(val, symbol, str(val)))
    return QgsCategorizedSymbolRenderer(field_name, categories)


def set_labeling(layer, field_name, format_type='decimal'):
    """정수/소수점 라벨링"""
    from qgis.core import QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat
    from PyQt5.QtGui import QColor

    settings = QgsPalLayerSettings()
    text_format = QgsTextFormat()
    text_format.setSize(10)
    text_format.setColor(QColor("black"))
    settings.setFormat(text_format)
    expression = f"format_number( \"{field_name}\", {0 if format_type == 'integer' else 3})"
    settings.fieldName = expression
    settings.isExpression = True
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def export_layer_to_image(layer, output_path, scale=None):
    """레이어를 이미지로 저장"""
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob, QgsProject
    from PyQt5.QtCore import QSize, QEventLoop
    from PyQt5.QtGui import QColor

    project = QgsProject.instance()
    project.addMapLayer(layer)
    map_settings = QgsMapSettings()
    map_settings.setLayers([layer])
    map_settings.setBackgroundColor(QColor(255, 255, 255))
    extent = layer.extent()
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
    QgsProject.instance().removeAllMapLayers()


def main():
    print("QGIS 자동화 스크립트 실행 시작...")
    setup_qgis_environment()

    from qgis.core import (QgsApplication, QgsVectorLayer, QgsRasterLayer,
                           QgsProject, QgsSymbol, QgsGraduatedSymbolRenderer,
                           QgsStyle, QgsClassificationEqualInterval)
    from qgis.analysis import QgsNativeAlgorithms
    from processing.core.Processing import Processing
    import processing

    qgs = QgsApplication([], False)
    qgs.initQgis()
    Processing.initialize()
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    for folder in [OUTPUT_IMAGE_FOLDER, OUTPUT_CSV_FOLDER, OUTPUT_TEMP_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    cv_data, growth_data = [], []
    vra_files = glob.glob(os.path.join(VRA_FOLDER, '*_VRA_Rx.tif'))

    if not vra_files:
        print(f"[오류] '{VRA_FOLDER}'에 VRA 파일이 없습니다.")
        return

    print(f"\n총 {len(vra_files)}개 필지 처리 시작...")

    for vra_path in vra_files:
        field_code = os.path.basename(vra_path).split('_')[0]
        print(f"\n▶ [{field_code}] 처리 중...")

        try:
            # --- 1단계: 벡터 변환 및 스타일 적용 ---
            vector_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vector.gpkg")
            processing.run("gdal:polygonize", {
                'INPUT': vra_path, 'BAND': 1, 'FIELD': 'DN',
                'EIGHT_CONNECTEDNESS': False, 'OUTPUT': vector_path
            })

            # 좌표계 통일
            reproject_path = os.path.join(OUTPUT_TEMP_FOLDER, f"{field_code}_vector_4326.gpkg")
            processing.run("native:reprojectlayer", {
                'INPUT': vector_path, 'TARGET_CRS': TARGET_CRS, 'OUTPUT': reproject_path
            })

            layer = QgsVectorLayer(reproject_path, f"{field_code}_vra", "ogr")
            layer.startEditing()
            layer.deleteFeatures([f.id() for f in layer.getFeatures() if f['DN'] == 0])
            layer.commitChanges()

            layer.setRenderer(get_categorized_renderer(layer, "DN"))
            set_labeling(layer, "DN", "integer")
            export_layer_to_image(layer, os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA.png"))
            print("   [성공] 1단계 완료")

            # --- 2단계: Before 구역통계 ---
            before_tif = find_data_file(field_code, BEFORE_FOLDER)
            processing.run("native:zonalstatisticsfb", {
                'INPUT': layer, 'INPUT_RASTER': before_tif, 'BAND': 1,
                'COLUMN_PREFIX': 'before_', 'STATISTICS': [2, 3, 5, 6, 7]
            })
            set_labeling(layer, "before_mean", "decimal")
            export_layer_to_image(layer, os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_before.png"))
            print("   [성공] 2단계 완료")

            # --- 3단계: After 구역통계 ---
            after_tif = find_data_file(field_code, AFTER_FOLDER)
            processing.run("native:zonalstatisticsfb", {
                'INPUT': layer, 'INPUT_RASTER': after_tif, 'BAND': 1,
                'COLUMN_PREFIX': 'after_', 'STATISTICS': [2, 3, 5, 6, 7]
            })

            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            ramp = QgsStyle.defaultStyle().colorRamp('Spectral')
            renderer = QgsGraduatedSymbolRenderer('after_mean')
            renderer.setSourceSymbol(symbol)
            renderer.setSourceColorRamp(ramp)
            renderer.updateClasses(layer, QgsClassificationEqualInterval(), 5)
            layer.setRenderer(renderer)

            set_labeling(layer, "after_mean", "decimal")
            export_layer_to_image(layer, os.path.join(OUTPUT_IMAGE_FOLDER, f"{field_code}_VRA_after.png"))
            print("   [성공] 3단계 완료")

            # --- 4단계: CSV 데이터 생성 ---
            gdf = gpd.read_file(reproject_path)
            before_std, after_std = gdf['before_mean'].std(), gdf['after_mean'].std()
            cv_data.append({'필지코드': field_code, 'before_mean_cv': before_std, 'after_mean_cv': after_std})

            for dn, grp in gdf.groupby('DN'):
                before_mean, after_mean = grp['before_mean'].mean(), grp['after_mean'].mean()
                vi_rate = ((after_mean - before_mean) / before_mean * 100) if before_mean else 0
                growth_data.append({
                    '필지코드': field_code, 'DN_Group': dn,
                    'before_mean_result': before_mean,
                    'after_mean_result': after_mean,
                    'vi_rate': vi_rate
                })
            print("   [성공] 4단계 CSV 집계 완료")

        except Exception as e:
            print(f"   [오류] {field_code} 처리 중 예외 발생: {e}")
            continue

    # --- 전체 CSV 저장 ---
    pd.DataFrame(cv_data).to_csv(os.path.join(OUTPUT_CSV_FOLDER, '변량시비 전후 표준편차.csv'),
                                index=False, encoding='utf-8-sig')
    pd.DataFrame(growth_data).to_csv(os.path.join(OUTPUT_CSV_FOLDER, '변량시비 전후 생육 변화량 확인.csv'),
                                    index=False, encoding='utf-8-sig')

    print("\n✅ 모든 필지 자동처리 완료!")
    qgs.exitQgis()


if __name__ == '__main__':
    main()
