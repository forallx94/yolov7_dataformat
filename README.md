# yolov7 Data Format

[makesense.ai](https://www.makesense.ai/) 에서 라벨링된 데이터를 이용  
[yolov7](https://github.com/WongKinYiu/yolov7) 학습용 데이터로 변경하는 코드


## makesense 데이터 구조

```
images
 - class 1
   - images 1
   - images 2
   - ...
 - class 2
 - ...
labels
 - class 1
   - txt 1
   - txt 2
   - ...
 - class 2
 - ...
```

## 코드 진행 후 결과 구조

```
train
 - images
 - labels
test 
 - images
 - labels
```

이후 labels.txt 파일을 이용하여 data.yaml 생성 필요

data.yaml 구조

```
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 3
names: ['head', 'helmet', 'person']
```