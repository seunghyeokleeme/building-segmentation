import os
import tarfile

def extract_tar(tar_path, extract_to_path):
    """tar 압축풀기"""
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to_path)
    print(f"{extract_to_path} 경로에 압축을 해제합니다.")

# 데이터 읽어오는 함수
dir_data = './xbd'
save_data = './xbd'

if not os.path.exists(save_data):
    os.makedirs(save_data)

# extract_tar(os.path.join(dir_data, 'train_images_labels_targets.tar'), save_data)
# extract_tar(os.path.join(dir_data, 'hold_images_labels_targets.tar'), save_data)
# extract_tar(os.path.join(dir_data, 'test_images_labels_targets.tar'), save_data)
# extract_tar(os.path.join(dir_data, 'tier3.tar'), save_data)

lst_train_images = [f for f in os.listdir(os.path.join(save_data, 'train', 'images')) if f.endswith('pre_disaster.png')]
lst_train_targets = [f for f in os.listdir(os.path.join(save_data, 'train', 'targets')) if f.endswith('pre_disaster_target.png')]
lst_hold_images = [f for f in os.listdir(os.path.join(save_data, 'hold', 'images')) if f.endswith('pre_disaster.png')]
lst_hold_targets = [f for f in os.listdir(os.path.join(save_data, 'hold', 'targets')) if f.endswith('pre_disaster_target.png')]
lst_test_images = [f for f in os.listdir(os.path.join(save_data, 'test', 'images')) if f.endswith('pre_disaster.png')]
lst_test_targets = [f for f in os.listdir(os.path.join(save_data, 'test', 'targets')) if f.endswith('pre_disaster_target.png')]
lst_tier_images = [f for f in os.listdir(os.path.join(save_data, 'tier3', 'images')) if f.endswith('pre_disaster.png')]
lst_tier_labels = [f for f in os.listdir(os.path.join(save_data, 'tier3', 'labels')) if f.endswith('pre_disaster.json')]

lst_train_images.sort()
lst_train_targets.sort()
lst_hold_images.sort()
lst_hold_targets.sort()
lst_test_images.sort()
lst_test_targets.sort()
lst_tier_images.sort()
lst_tier_labels.sort()

print(f"Number of training images: {len(lst_train_images)}")
print(f"Number of training targets: {len(lst_train_targets)}")
print(f"Number of hold images: {len(lst_hold_images)}")
print(f"Number of hold targets: {len(lst_hold_targets)}")
print(f"Number of test images: {len(lst_test_images)}")
print(f"Number of test targets: {len(lst_test_targets)}")
print(f"Number of tier images: {len(lst_tier_images)}")
print(f"Number of tier labels: {len(lst_tier_labels)}")

print(lst_train_images[0])
print(lst_train_targets[0])
print(lst_hold_images[0])
print(lst_hold_targets[0])
print(lst_test_images[0])
print(lst_test_targets[0])
print(lst_tier_images[0])
print(lst_tier_labels[0])