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

-------------------------------------------------------------------------------------

treatment -> 5:5
survive ->편향

>> smote 사용


데이터 불균형 참고
https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model
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