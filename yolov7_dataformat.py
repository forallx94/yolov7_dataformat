import os
import shutil
from glob import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

output_path = 'output'

txt_path = [Path(i) for i in glob('./labels/*/*') if Path(i).suffix == '.txt'] # 라벨링 데이터 중 txt 파일 추출
txt_path = [i for i in txt_path if i.stem != 'labels'] # labels 파일 제외
# 라벨 내역으로 데이터 프레임 생성

df = pd.DataFrame(
    [(i.parts[-2], i.parts[-1], i.stem,) for i in txt_path],
    columns=['class', 'file_name', 'base_name']
    )

x_train, x_valid = train_test_split(df, stratify=df['class'], test_size= 0.2)
os.makedirs( f'./{output_path}/train/images',exist_ok=True)
os.makedirs( f'./{output_path}/train/labels',exist_ok=True)
os.makedirs( f'./{output_path}/valid/images',exist_ok=True)
os.makedirs( f'./{output_path}/valid/labels',exist_ok=True)

def image_file_check(class_,base_name):
    '''
    라벨 파일 기준으로 이미지 파일의 확장자를 찾음
    라벨 파일과 대응되는 이미지 파일을 못찾을 경우 에러 발생
    '''
    for suffix_ in ['.JPG', '.jpeg', '.jpg', '.png', '.webp']:
        if os.path.isfile(f'./images/{class_}/{base_name}{suffix_}'):
            return f'./images/{class_}/{base_name}{suffix_}'
    raise Exception(f"{class_}/{base_name} Not exist")

duplicate_checker = dict({})
for _, (class_, _, base_name) in x_train.iterrows():

    if base_name in duplicate_checker: # 기존 파일명과 중복 될 시
        # 이미지 복사
        shutil.copy2(
            image_file_check(class_,base_name),
            f'./{output_path}/train/images/{base_name}_{duplicate_checker[base_name]}.jpg'
        )
        # 라벨 복사
        shutil.copy2(
            f'./labels/{class_}/{base_name}.txt',
            f'./{output_path}/train/labels/{base_name}_{duplicate_checker[base_name]}.txt'
        )
        duplicate_checker[base_name] += 1
    else: # 중복이 없을 시
        # 이미지 복사
        shutil.copy2(
            image_file_check(class_,base_name),
            f'./{output_path}/train/images/{base_name}.jpg'
        )
        # 라벨 복사
        shutil.copy2(
            f'./labels/{class_}/{base_name}.txt',
            f'./{output_path}/train/labels/{base_name}.txt'
        )
        duplicate_checker[base_name] = 0

duplicate_checker = dict({})
for _, (class_, _, base_name) in x_valid.iterrows():

    if base_name in duplicate_checker: # 기존 파일명과 중복 될 시
        # 이미지 복사
        shutil.copy2(
            image_file_check(class_,base_name),
            f'./{output_path}/valid/images/{base_name}_{duplicate_checker[base_name]}.jpg'
        )
        # 라벨 복사
        shutil.copy2(
            f'./labels/{class_}/{base_name}.txt',
            f'./{output_path}/valid/labels/{base_name}_{duplicate_checker[base_name]}.txt'
        )
        duplicate_checker[base_name] += 1
    else: # 중복이 없을 시
        # 이미지 복사
        shutil.copy2(
            image_file_check(class_,base_name),
            f'./{output_path}/valid/images/{base_name}.jpg'
        )
        # 라벨 복사
        shutil.copy2(
            f'./labels/{class_}/{base_name}.txt',
            f'./{output_path}/valid/labels/{base_name}.txt'
        )
        duplicate_checker[base_name] = 0
