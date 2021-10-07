----------------------------------------------------------------------
* 제시하는 유전자 후보
----------------------------------------------------------------------

G211, G179, G80, G27, G147, G139, G242, G264, G290, G130


----------------------------------------------------------------------
* 첨부 파일은 아래와 같습니다.
----------------------------------------------------------------------

1. 설명 문서
    (디지털헬스 헤커톤 보고서-의학으학으악!!!.docx)

2. Code 폴더
    - CSV 파일
        (Clinical_Variables.csv, Genetic_alterations.csv, Survival_time_event.csv, Treatment.csv, Label.csv)

    - Deep Learning 관련 코드 파일
        (dataloader.py, main.py, model.py, test.py, train.py, util.py)

    - 새로운 Label CSV 생성 관련 코드 파일
        (Label.py)

    - Machine Learning 관련 코드 파일
        (ML_Test.py)

3. Results 폴더
(Genetic Variables Frequency Histogram of Random Set Machine Learning _ LightGBM.png,
Genetic Variables Frequency Histogram of Random Set Machine Learning _ Random Forest.png,
Random_Set_Deep_Learning_Results.txt,
Random_Set_Machine_Learning_LightGBM_Results.txt,
Random_Set_Machine_Learning_RandomForeset_Results.txt,
Selected_Set_Deep_Learning_Results.txt,
Top10_Genetic_Candidate.txt)


----------------------------------------------------------------------
* 코드 사용 방법
----------------------------------------------------------------------

1. OS 검색창에서 
"cmd" 검색 - 윈도우, 
"terminal" 검색 - 리눅스, MacOS, 

2. 환경 설정 
Deep Learning: PyTorch, Tensorboard 설치
Machine Learning: lightgbm, matplotlib, scikit-learn 설치
(환경 설정 문제 있을시 바로 알려주세요!)

3. Label.csv 생성 방법
python Label.py

4. Deep Learning Model로 Genetic Variables 300개에 대해서 학습
python main.py

5. Deep Learning Model로 Genetic Variables 10개에 대해서 학습
python main.py --selected_genetic G211,G179,G80,G27,G147,G139,G242,G264,G290,G130
(주의! Genetic Variables가 10개가 아니면 에러남. ',' 붙여야되고 띄어쓰기 있으면 안됨)

6. Machine Learning Model Test 방법
python ML_Test.py

