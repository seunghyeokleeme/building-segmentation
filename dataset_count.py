import os

# 데이터 읽어오는 함수
dir_data = './datasets'
# dir_data = './xbd'

lst_train_images = [f for f in os.listdir(os.path.join(dir_data, 'train', 'images')) if f.endswith('pre_disaster.png')]
lst_train_targets = [f for f in os.listdir(os.path.join(dir_data, 'train', 'targets')) if f.endswith('pre_disaster_target.png')]
lst_hold_images = [f for f in os.listdir(os.path.join(dir_data, 'hold', 'images')) if f.endswith('pre_disaster.png')]
lst_hold_targets = [f for f in os.listdir(os.path.join(dir_data, 'hold', 'targets')) if f.endswith('pre_disaster_target.png')]
lst_test_images = [f for f in os.listdir(os.path.join(dir_data, 'test', 'images')) if f.endswith('pre_disaster.png')]
lst_test_targets = [f for f in os.listdir(os.path.join(dir_data, 'test', 'targets')) if f.endswith('pre_disaster_target.png')]
# lst_tier_images = [f for f in os.listdir(os.path.join(dir_data, 'tier3', 'images')) if f.endswith('pre_disaster.png')]
# lst_tier_labels = [f for f in os.listdir(os.path.join(dir_data, 'tier3', 'labels')) if f.endswith('pre_disaster.json')]
# lst_tier_targets = [f for f in os.listdir(os.path.join(dir_data, 'tier3', 'targets')) if f.endswith('pre_disaster_target.png')]

lst_train_images.sort()
lst_train_targets.sort()
lst_hold_images.sort()
lst_hold_targets.sort()
lst_test_images.sort()
lst_test_targets.sort()
# lst_tier_images.sort()
# lst_tier_labels.sort()
# lst_tier_targets.sort()

print(f"Number of training images: {len(lst_train_images)}")
print(f"Number of training targets: {len(lst_train_targets)}")
print(f"Number of hold images: {len(lst_hold_images)}")
print(f"Number of hold targets: {len(lst_hold_targets)}")
print(f"Number of test images: {len(lst_test_images)}")
print(f"Number of test targets: {len(lst_test_targets)}")
# print(f"Number of tier images: {len(lst_tier_images)}")
# print(f"Number of tier labels: {len(lst_tier_labels)}")
# print(f"Number of tier targets: {len(lst_tier_targets)}")

print(lst_train_images[0])
print(lst_train_targets[0])
print(lst_hold_images[0])
print(lst_hold_targets[0])
print(lst_test_images[0])
print(lst_test_targets[0])
# print(lst_tier_images[0])
# print(lst_tier_labels[0])
# print(lst_tier_targets[0])
