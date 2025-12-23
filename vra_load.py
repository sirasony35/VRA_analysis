from PIL import Image
import numpy as np
import pandas as pd
import os
import math


def analyze_vra_with_colors(file_path):
    """
    WGS 84 타원체 정밀 면적 계산, 총 살포량 산출 및 등급별 색상을 부여합니다.
    """
    if not os.path.exists(file_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
        return None

    try:
        with Image.open(file_path) as img:
            # 1. 지리 정보 및 해상도 추출
            pixel_scale = img.tag_v2.get(33550)
            tie_points = img.tag_v2.get(33922)

            if not pixel_scale or not tie_points:
                gsd_x = gsd_y = 1.0
            else:
                deg_x, deg_y = pixel_scale[0], pixel_scale[1]
                ref_lat = tie_points[4]
                rad_lat = math.radians(ref_lat)

                a = 6378137.0
                f = 1 / 298.257223563
                e2 = 2 * f - f ** 2
                denom = (1 - e2 * (math.sin(rad_lat) ** 2))
                m_per_deg_lat = (math.pi * a * (1 - e2)) / (180 * (denom ** 1.5))
                m_per_deg_lon = (math.pi * a * math.cos(rad_lat)) / (180 * math.sqrt(denom))
                gsd_x, gsd_y = deg_x * m_per_deg_lon, deg_y * m_per_deg_lat

            # 2. 데이터 카운팅
            data = np.array(img)
            unique, counts = np.unique(data, return_counts=True)

            # 3. 면적 및 살포량 계산
            pixel_area = gsd_x * gsd_y
            areas_ha = (counts * pixel_area) / 10000
            total_amounts = unique * areas_ha

            # 4. 결과 데이터프레임 구성
            df = pd.DataFrame({
                'DN_Value': unique,
                'Area_ha': areas_ha,
                'Total_Amount': total_amounts
            })

            # 5. [추가] DN_Value 기준 색상 매핑 로직
            # 배경(0)을 제외한 실제 등급만 추출하여 내림차순 정렬
            target_dns = sorted([val for val in unique if val > 0], reverse=True)

            # 높은 순서대로 부여할 색상 리스트
            color_list = ['Red', 'Orange', 'Beige', 'LightGreen', 'Green']
            color_map = {0.0: 'None'}  # 배경은 색상 없음

            for i, val in enumerate(target_dns):
                if i < len(color_list):
                    color_map[val] = color_list[i]
                else:
                    color_map[val] = 'Green'  # 5단계 이후는 모두 초록색 처리

            df['Color'] = df['DN_Value'].map(color_map)

            # 6. 최종 리포트 출력
            print(f"\n" + "=" * 85)
            print(f" [VRA 최종 분석 보고서 - 시각화 등급 포함] ")
            print(f" 파일명: {os.path.basename(file_path)}")
            print(f"=" * 85)

            pd.options.display.float_format = '{:.4f}'.format
            # 컬럼 순서 조정
            df = df[['DN_Value', 'Color', 'Area_ha', 'Total_Amount']]
            print(df.to_string(index=False))
            print(f"=" * 85)

            work_df = df[df['DN_Value'] > 0]
            print(f"1. 전체 작업 면적: {work_df['Area_ha'].sum():.4f} ha")
            print(f"2. 전체 비료/약제 소요량: {work_df['Total_Amount'].sum():.2f} (kg/L)")
            print(f"3. 등급 기준: 빨간색(최대 살포) -> 초록색(최소 살포)")
            print(f"=" * 85)

            return df

    except Exception as e:
        print(f"[오류] 발생: {e}")
        return None


if __name__ == "__main__":
    target_file = "data/sc/sc_vra_data/SC03_GNDVI - 경계 대상 작업_Rx.tif"
    analyze_vra_with_colors(target_file)