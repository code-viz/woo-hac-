selected_genetic = ['G211', 'G264', 'G179', 'G27', 'G147', 'G139', 'G242', 'G80', 'G263', 'G290']
genetic_table = genetic_table[selected_genetic]


1. Genetic 10개를 뽑은 방법
    => Deep Learning Model
2. 뽑힌 Genetic 10개가 좋다는 것을 증명
    => Machine Learning Model


DL로 뽑힌 Genetic 10개
랜덤으로 뽑은 Genetic 10개
DL로 뽑힌 Genetic 10개 -> 1세트
랜덤으로 뽑은 Genetic 10개 -> 100세트


1. Train Acc가 100이 나오는 경우
2. Train Acc가 100이 안나오는 경우


2번에 대해서 실험하고 방법 쓰기


1. DL로 뽑힌 Genetic 10개 -> 1세트
2. 랜덤으로 뽑은 Genetic 10개 -> 100세트


2번 세트에서 1번 세트를 이기는 경우
이긴 2번 Genetic 세트5개 정도 저장 => 3번
3번과 1번을 여러번 돌려서 1번이 더 많이 이긴다는 것을 보여준다.

--------------------------------------------------------------------

DL로 Top Genetic 20개 
DL로 Top Genetic 30개 
DL로 Top Genetic 40개 
DL로 Top Genetic 50개 

--------------------------------------------------------------------

DL로 뽑힌 Top Genetic 10개 -> 1세트
랜덤으로 뽑은 Genetic 10개 -> 100세트

DL Set이 Random Set보다 좋다는 것을 증명
1. Train Acc가 100이 나오는 경우
2. Train Acc가 100이 안나오는 경우

Random Set에서 DL Set를 이기는 경우
DL Set을 이긴 Random Set 5개 정도 저장 => Win Random Set
Win Random Set과 DL Set을 여러번 돌려서 평균적으로 DL Set이 더 많이 이긴다는 것을 보여준다.

--------------------------------------------------------------------

코드 돌리는 가이드 라인도 같이 작성
코드 깔끔하게


--------------------------------------------------------------------










-------------------------------------------------------------------------------------

암 -> 유전자 변이 -> 치료를 했을 때 -> 효과적 -> 살아남았다. or 오래 살았다(?)

-------------------------------------------------------------------------------------

머신러닝 학습이 완료된 모델에
전체 환자 데이터를 가지고
유전자 변이 데이터들을 수정해서
모델에 입력으로 넣어줬을 때
결과값이 바뀌는 유전자 변이를 찾는다.
(Survive -> Die or Die -> Survive)

-------------------------------------------------------------------------------------

영향력이 있는 데이터가 뭔지 ml에 분석기능?


https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
https://talkingaboutme.tistory.com/entry/ML-TIP-Logistic-Regression-Feature-importance
https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model

-------------------------------------------------------------------------------------

treatment -> 5:5
survive ->편향

>> f1 score 확인
>> smote 사용


데이터 불균형 참고
https://john-analyst.medium.com/smote%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-5ab674ef0b32
https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/
https://machinelearningmastery.com/multi-class-imbalanced-classification/

-------------------------------------------------------------------------------------

1. 모델 설계
2. Genetic 10을 어떻게 뽑을 것인가
치료의 효과를 증가시키는 인과관계가 있는 유전자
=> 치료를 해서 살아남은 사람
=> Treatment & Survive

-------------------------------------------------------------------------------------

치료o + 사망x -> class 0 (48명)
나머지 경우 -> class 1

BaseModel을 만들고
Smote를 이용하여 데이터를 생성
BaseModel의 성능이 떨어지지 않는 데이터들을 수집
Treatment & Survive 데이터들을 분석

-------------------------------------------------------------------------------------

1. Base Model 학습
    (Treatment & Survive : 0 / 나머지 : 1)
2. SMOTE 데이터 생성
3. SMOTE 데이터를 Base Model로 Inference
4. Base Model이 Treatment & Survive라고 예측하여 맞춘 SMOTE 데이터들을 수집
5. 원본 Treatment & Survive 데이터에 SMOTE 데이터들을 차례차례 추가해보면서 변동성이 큰 Genetic 변수들을 찾아낸다.
(가정 : 변수값이 변화해도 결과값에 영향이 없다는 것은 결과 예측을 위한 중요한 변수가 아니다.)
6. 원본 데이터에서 해당 Genetic 변수들을 제거한다.
7. 수정된 데이터로 1~6까지 다시 반복한다.

-------------------------------------------------------------------------------------