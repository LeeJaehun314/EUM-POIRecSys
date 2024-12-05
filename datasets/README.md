# 데이터셋 다운로더 사용법
다운로더는 Linux/Mac Shell 환경에서만 동작합니다. Windows에서 사용하시려면 WSL 상에서 사용해 주세요. 다운로더는 AIHUB 로부터 데이터를 다운로드 받아 압축을 풉니다.

1. 다음 명령어를 통해 데이터를 다운로드 합니다.

    ./dataset_downloader AIHUB_ID 'AIHUB_PW'


# 다운로드 세부 단계
다운로더에 문제가 있거나 직접 명령어를 통해 다운로드하시려는 경우 아래 단계를 따라서 다운로드 할 수 있습니다.  
*참고. https://www.aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100*

1.  (Windows인 경우만) 다음 명령어로 WSL ubuntu 설치

    ubuntu

2. (Windows인 경우만) 프로젝트 폴더에서 다음 명령어를 통해 ubuntu shell 실행

    ubuntu run 

3. 다음 명령어로 aihubshell 다운로드

    curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do

4. aihub 로그인 정보 설정  

    export AIHUB_ID=로그인_이메일  
    export AIHUB_PW='로그인_비밀번호'

5. 2023 수도권 데이터셋(datasetkey: 71776) 다운로드 (이미지 제외, GPS 데이터 제외)

    ./aihubshell -mode d -datasetkey 71776 -filekey 539782,539784,539787

6. 2023 제주도 데이터셋(datasetkey: 71780) 다운로드 (이미지 제외, GPS 데이터 제외)

    ./aihubshell -mode d -datasetkey 71780 -filekey 541665,541667,541670

7. 데이터 압축 풀기

    find . -type f -name "*.zip" -exec sh -c 'for zip_file; do folder_name="${zip_file%.zip}"; echo "Unzipping $zip_file to $folder_name..."; mkdir -p "$folder_name" && 7z x "$zip_file" -o"$folder_name"; done' _ {} +

8. (필요시) zip 파일 삭제

    find . -type f -name "*.zip" -delete

# 수도권 데이터셋 구조

    └─145.국내 여행로그 데이터_수도권_2차년도
        └─3.개방데이터
            └─1.데이터
                ├─Other
                │  └─Other.zip | 386 MB | 539782
                ├─Training
                │  ├─01.원천데이터
                │  │  └─TS_photo.zip | 70 GB | 539783
                │  └─02.라벨링데이터
                │      ├─TL_csv.zip | 3 MB | 539784
                │      └─TL_gps_data.zip | 90 MB | 539785
                ├─Validation
                │  ├─01.원천데이터
                │  │  └─VS_photo.zip | 9 GB | 539786
                │  └─02.라벨링데이터
                │      ├─VL_csv.zip | 489 KB | 539787
                │      └─VL_gps_data.zip | 11 MB | 539788
                └─Sublabel
                    └─SbL.zip | 15 GB | 549764

# 제주도 데이터셋 구조

    └─148.국내 여행로그 데이터_제주도 및 도서지역_2차년도
        └─3.개방데이터
            └─1.데이터
                ├─Other
                │  └─Other.zip | 386 MB | 541665
                ├─Training
                │  ├─01.원천데이터
                │  │  └─TS_photo.zip | 69 GB | 541666
                │  └─02.라벨링데이터
                │      ├─TL_csv.zip | 5 MB | 541667
                │      └─TL_gps_data.zip | 236 MB | 541668
                ├─Validation
                │  ├─01.원천데이터
                │  │  └─VS_photo.zip | 8 GB | 541669
                │  └─02.라벨링데이터
                │      ├─VL_csv.zip | 780 KB | 541670
                │      └─VL_gps_data.zip | 29 MB | 541671
                └─Sublabel
                    └─SbL.zip | 15 GB | 549769
