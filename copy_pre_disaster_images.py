import os
import shutil

# 원본 datasets 폴더 경로 (예시)
datasets_dir = "./xbd"  # 실제 경로로 수정

# 대상 폴더 경로 (복사될 폴더의 부모 디렉터리)
dest_dir = "."  # 현재 디렉터리(또는 원하는 부모 경로)로 지정

def create_dataset_structure(base_path='.'):
    """
    주어진 base_path 하위에 datasets 폴더 및
    hold, test, train 폴더와 각각의 images, targets 폴더를 생성합니다.
    
    Parameters:
    -----------
    base_path: str
        폴더 구조를 생성할 기준 경로 (예: 현재 디렉터리 '.')
    """
    # 생성할 폴더 구조 정의 ("datasets" 접두사가 포함됨)
    folders = [
        "datasets/hold/images",
        "datasets/hold/targets",
        "datasets/test/images",
        "datasets/test/targets",
        "datasets/train/images",
        "datasets/train/targets",
    ]
    
    for folder in folders:
        target_path = os.path.join(base_path, folder)
        os.makedirs(target_path, exist_ok=True)
        print(f"Created: {target_path}")

# 대상 폴더 내에 "datasets" 폴더가 없으면 생성
if not os.path.exists(os.path.join(dest_dir, "datasets")):
    create_dataset_structure(dest_dir)

def copy_pre_disaster_images(datasets_dir, dest_dir, include_tier3=False):
    """
    datasets_dir(원본)에서 dest_dir(부모 디렉터리) 내의 datasets 폴더로
    pre_disaster 이미지를 복사하는 함수.
    tier3 폴더를 train 데이터에 포함할지 여부를 include_tier3 매개변수로 결정합니다.
    
    Parameters:
    -----------
    datasets_dir: str
        원본 데이터셋 폴더 경로 (예: './xbd')
    dest_dir: str
        대상 폴더의 부모 디렉터리 (예: '.' -> 생성되는 폴더는 './datasets')
    include_tier3: bool
        True이면 tier3 폴더 내 파일을 train 데이터로 복사합니다.
        False이면 tier3 폴더는 무시합니다.
    """
    # hold, test, train 폴더 내의 images 폴더 순회
    for split in ['hold', 'test', 'train']:
        source_dir = os.path.join(datasets_dir, split, 'images')
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                if filename.endswith("pre_disaster.png"):
                    src_path = os.path.join(source_dir, filename)
                    
                    image_path = os.path.join(dest_dir, 'datasets', split, 'images', filename)
                    target_path = os.path.join(
                        dest_dir, 'datasets', split, 'targets',
                        filename.split('.png')[0] + '_target.png'
                    )
                    
                    shutil.copy2(src_path, image_path)
                    
                    target_src_path = os.path.join(
                        datasets_dir, split, 'targets',
                        filename.split('.png')[0] + '_target.png'
                    )
                    shutil.copy2(target_src_path, target_path)
                    
                    print(f"[{split}] {filename} 파일 복사 완료.")
        else:
            print(f"{source_dir} 폴더가 존재하지 않습니다.")
    
    # tier3 폴더 처리: 포함 여부에 따라 train 데이터로 복사
    if include_tier3:
        tier3_dir = os.path.join(datasets_dir, 'tier3', 'images')
        if os.path.exists(tier3_dir):
            for filename in os.listdir(tier3_dir):
                if filename.endswith("pre_disaster.png"):
                    src_path = os.path.join(tier3_dir, filename)
                    
                    image_path = os.path.join(dest_dir, 'datasets', 'train', 'images', filename)
                    target_path = os.path.join(
                        dest_dir, 'datasets', 'train', 'targets',
                        filename.split('.png')[0] + '_target.png'
                    )
                    
                    shutil.copy2(src_path, image_path)
                    
                    target_src_path = os.path.join(
                        datasets_dir, 'tier3', 'targets',
                        filename.split('.png')[0] + '_target.png'
                    )
                    shutil.copy2(target_src_path, target_path)
                    
                    print(f"[tier3 -> train] {filename} 파일 복사 완료.")
        else:
            print(f"{tier3_dir} 폴더가 존재하지 않습니다.")

# 사용 예시
if __name__ == "__main__":
    # tier3 폴더를 train 데이터에 포함시키려면 include_tier3 값을 True로 설정합니다.
    copy_pre_disaster_images(
        datasets_dir=datasets_dir,
        dest_dir=dest_dir,
        include_tier3=False
    )
