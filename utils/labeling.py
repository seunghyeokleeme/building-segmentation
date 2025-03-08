import json
import os
import numpy as np
from shapely import wkt
from PIL import Image, ImageDraw

save_dir = "./xbd/tier3/targets"
label_dir = "./xbd/tier3/labels"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_mask(dir_label, save_dir):
    lst_label = [f for f in os.listdir(dir_label) if f.endswith('pre_disaster.json')]
    
    for label_filename in lst_label:
        # JSON 파일 읽기
        with open(os.path.join(dir_label, label_filename), "r") as f:
            data = json.load(f)

        # "xy" 필드에서 건물(feature_type == "building") 객체 추출
        xy_features = data["features"]["xy"]
        building_features = [feat for feat in xy_features if feat["properties"].get("feature_type") == "building"]

        # 출력 이미지 크기: 1024x1024, 초기 마스크(배경=0)
        img_size = 1024
        mask = Image.new("L", (img_size, img_size), 0)

        if building_features:
            # WKT 문자열을 shapely 다각형으로 변환
            polygons = [wkt.loads(feat["wkt"]) for feat in building_features]
            
            # 전체 건물 영역의 좌표 바운딩 박스 계산 (건물 데이터가 있을 때만)
            bounds = [poly.bounds for poly in polygons]
            min_x = min(b[0] for b in bounds)
            min_y = min(b[1] for b in bounds)
            max_x = max(b[2] for b in bounds)
            max_y = max(b[3] for b in bounds)
            
            draw = ImageDraw.Draw(mask)

            def transform_coords(coords, min_x, min_y, max_x, max_y, img_size):
                """
                주어진 좌표를 전체 건물 영역의 바운딩 박스(min_x, min_y, max_x, max_y)에 대해
                1024x1024 픽셀 좌표로 선형 변환합니다.
                """
                transformed = []
                for x, y in coords:
                    # x 좌표 선형 변환
                    px = (x - min_x) / (max_x - min_x) * (img_size - 1)
                    # y 좌표 단순 선형 변환
                    py = (y - min_y) / (max_y - min_y) * (img_size - 1)
                    transformed.append((px, py))
                return transformed

            # 각 건물 다각형을 좌표 변환 후 이미지에 그리기 (건물 영역은 1로 채움)
            for poly in polygons:
                coords = list(poly.exterior.coords)
                new_coords = transform_coords(coords, min_x, min_y, max_x, max_y, img_size)
                draw.polygon(new_coords, outline=1, fill=1)
            print(f"{label_filename}: 건물 데이터 마스킹 생성!")   
        else:
            # 건물 데이터가 없는 경우: 기본적으로 모두 0인 마스크를 그대로 사용
            print(f"{label_filename}: 건물 데이터가 없습니다.")

        # 결과 마스크 PNG 저장 (파일명: 원본 이름에서 .json 제거 후 _target.png 추가)
        mask.save(os.path.join(save_dir, label_filename.split(".json")[0] + "_target.png"))


save_mask(label_dir, save_dir)