import os
import shutil

# 원본 datasets 폴더 경로 (예시)
datasets_dir = "./xbd"  # 실제 경로로 수정

# 대상 폴더 경로 (복사될 폴더)
dest_dir = "./datasets"  # 실제 경로로 수정

# 대상 폴더가 없으면 생성
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# hold, test, train 폴더 내의 image와 targets 폴더 순회
for split in ['hold', 'test', 'train', 'tier3']:
    for subfolder in ['images', 'targets']:
        source_dir = os.path.join(datasets_dir, split, subfolder)
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                if filename.endswith("pre_disaster.png"):
                    src_path = os.path.join(source_dir, filename)
                    if split == "tier3":
                        image_path = os.path.join(dest_dir,'train', subfolder, filename)
                        target_path = os.path.join(dest_dir,'train', 'targets', filename.split('.png')[0]+'_target.png')
                    else:
                        image_path = os.path.join(dest_dir,split, subfolder, filename)
                        target_path = os.path.join(dest_dir,split, 'targets', filename.split('.png')[0]+'_target.png')
                    # 파일 복사 (이동하고 싶다면 shutil.move(src_path, dst_path) 사용)
                    shutil.copy2(src_path, image_path)
                    shutil.copy2(os.path.join(datasets_dir, split, 'targets',filename.split('.png')[0]+'_target.png'), target_path)
                    print("파일 이동!")
        else:
            print(f"{source_dir} 폴더가 존재하지 않습니다.")
