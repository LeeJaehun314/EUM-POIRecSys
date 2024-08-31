# 필요한 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np
from surprise import SVD
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader
from collections import defaultdict
import pickle
import time

starttime = time.time()


dfj = pd.read_csv('./1.inputdata/tn_visit_area_info_H.csv')

dff = dfj.dropna(subset = ['LODGING_TYPE_CD'])  # 숙박 유형이 없는 변수 제거
dff = dff.dropna(subset = ['DGSTFN'])  # 평점이 없는 변수 제거

# 의미 없는 변수 제거
words_to_remove = ["집", "스타벅스", '치과','숙소','에어비앤비','약국','송해','케이크란',
                   '서울대공원','국민은행본점', '김포국제공항 국내선','육백마지기','서울고속버스터미널(경부)',
                  '김선생국수&꼬치', '스튜디오한사','압구정로데오거리','신세계푸드비스트로870',
                  '본동신동아아파트', 'Jjs house', '화랑활터', '로스트인홍콩','글레이셔박','부평현대더로프트',
                  '인천광역시 강화군 강화읍 관청리 713 강화유니버스', '인천광역시 강화군 강화읍 관청리 713 2층 강화유니버스',
                  '성신카센타', '송파법조타운푸르지오', '휴먼시아7단지아파트', '뉴밀레니엄타워', '숭실사이버대학교종로캠퍼스',
                  '라세', '금남멤버스오피스텔', '첫날 여행의 바베큐와 술한잔 즐기고 잘 펜션', '연천전곡리유적방문자센터',
                   '월미공원어을미센터','올림픽공원', '상천루','와석초등학교', '순화동THE#', '장원막국수',
                   '메트로타워', '충무정', '광교호반마을22단지아파트','아차산', '원오브어스', '영등포역파출소', '복포마을 별장',
                   '대박물류','산장관광지고객지원센터', '국수포차', '이루라책방',
                  '양지파인골프클럽', '달맞이광장', '전곡항마리나클럽하우스','대지빌딩','현충사입구', '코리아다이어트센타',
                  '태화강역', '수원역', '롯데월드어드벤처', '목포역', '평택역', '노숙인현장상담실', '홍성읍 고암리13', '적목리',
                  '서울역버스환승센터', 'E편한세상시티미사오피스텔', '서울고속버스터미널', '영진해수욕장', 'T.H 하우스',
                   '뉴골드프라자','춘천역사', '연천', '크레신빌딩', '래미안광교아파트', 'EX오일충주주유소', '아틴마루',
                   '프라자컨트리클럽', '산운마을2단지', '천주교전동교회', '용산역','홍익스포츠프라자', '르네상스시티프라자',
                   '에어비엔비 : Jane의 개인실','인천선린동공화춘', '포항터미널', '스튜디오161',
                   '평택비전지웰푸르지오아파트', '모곡밤벌유원지', '은영아파트/노인정',
                  '현대아이파크', '샤워실','플로어탑노트','더스테이힐링파크 내 알파와앵무', '마산버스터미널', '한강센트리움빌리지',
                  '선릉정릉', '루시하우스', '동국사','롯데백화점본점', '할머니댁', '죽향문화체험마을죽녹원', '사무실',
                  '신성프라자1', '롯데프리미엄아울렛', '대양빌딩', '천제연폭포매표소', '양평쉬자파크해몽', '이천테르메덴',
                  '대룡정', '제주중문대포해안주상절리대방문객센터', '소망빌딩', '성산일출봉', '광교호수공원', '대천해수욕장',
                  '상쾌한이비인후과의원', '한라비발디캠퍼스2차아파트', '광명한라비발디큐브', '덧정하우스', '맥도날드 상계DT점',
                  '세명하이테크', '가보자빌딩', '슬리피판다', '대훈빌딩', '뜰안어린이공원','뉴골드프라자','한국해양대학교 승선생활관',
                  'e편한세상추동공원2차아파트', '목감베르디움더프라임', '청정낚시터','산들래자연체험학교', '우리농산물유통센터',
                  '역전할머니맥주', '일산웨스턴돔점대양빌딩','국일관드림펠리스', '금화마을5단지아파트', '석모도 만남의광장'
                  '국민은행자산관리프라자', '경인아라뱃길','어반아지트',  '예단포항', '금성빌라상가', 'No.25','강남블루지움',
                  '구둔역','노원중교', '소바의온도', '북한산아이파크아파트', '경기도 수원시 팔달구 중동 49-3', '파티룸',
                  '해동용궁사', '노봉해변','엘스테이2차', '디스커버리서핑', '선박에서 숙박', '청태산 자연속으로',
                  '서울역울릉역', '연명 1길', '하나로마트 조정점','0도', '칠계재','함스베이커리', '청초대우아파트',
                  '김녕해수욕장', '드림유스타운', '동대구역', '동대구유성푸르나임/김건우산부인과', '부띠끄시티테라스3차오피스텔',
                  '부산광안대우아이빌', '부산광안대우아이빌오피스텔', '오락실','GS25송도비치점', 'GS25 제주함덕점',
                  '명주마지', 'Joy house', '플레아 드 블랑', '후포해수욕장',' 한국해양과학기술원 울릉도독도해양연구기지', '선창선착장',
                  '민박','백두대간생태수목원관리사무소', '염전해변캠핑장', '한양고시원','성보전', '즐거운동물병원',
                  '강릉원예농협하나로마트 초당점', '낙산해변', '송정해변', '근화동', '옴카페', '장흥주민복합센터', '서광프라자',
                  '리더스볼링센터', '뜨란채', '와이키키서프', '곤충전시관', '강릉시공항길30번길21', '레츠고카페',
                   '자면서 고속버스로 이동중', '무학아파트', '성원푸드몰', 'BROWN DOT HOTEL', '대송 문구사', '크루즈여객선객실',
                  '착한슈퍼', '스노클비치', '속초농협하나로마트', '가평휴게소 춘천방향', '고기백화점', '왕곡길',
                  '옴뷔', '서프홀릭 강릉점', '낙산도립공원', '부산역', '통영시 중앙로 175', '오산식당', '해파랑길 10코스',
                  '앙젤루스 소원테마파크', '한섬감성바닷길', '타워더모스트광안오피스텔', '안목해변', '순천농협파머스마켓',
                  '대진1리해변', '그리고게스트하우스', '에픽서프', '천진해변 주차장', '하조대해수욕장',
                   '봉산동1369업무시설(주식회사세명씨엔씨)', '파로스오피스텔', '안녕, 송도', '밀봄숲', '퍼니수제치킨',
                  '단독주택(다가구)', '롯데호텔롯데백화점롯데시네마', '드림센터', '속초세관기숙사', '씨스타빌딩',
                  '허가9425', '플라자CC설악', '큐브빌딩', '삼귀해안', '사동해수욕장', '영월 동강 시스타', '울릉군도동정류소',
                  '해동용궁사', '노봉해변', '디스커버리서핑', '엘스테이2차', '부모님 댁', '반월당역클래시아2차오피스텔',
                  '예다원', '강릉별당', '하우스못골', '시골민박', '상주문장대특산물직판장', '선비촌체험장', '이상원미술관',
                  'snug shelter', '무릉계곡명승지관리사무소', 'KT&G상상마당아트센터', '초량이바구길', '단양생태체육공원',
                  '대매물도 당금마을 폐교야영장', '석산산촌생태마을', '체크아웃 후 강원랜드 하이원리조트 내 카지노 구경',
                 '르컬렉티브 부산역', '스튜디오202', '런던브라우니', '경포대신도브래뉴로얄카운티아파트',
                   '금역당사당및종가(경상북도유형문화재제25호)', '초희전통차체험관', '대신펠리체','메르디앙', '유경식당',
                   '모찌호스텔', '욜로하', '기회송림유원지', '간성전통시장, 하나로마트', '똥강아지haus', '서천시장',
                   '진주소사이어트',
                   '황금연어공원', '시티오피스텔', '거창수승대', '캠핑', '구룡자동차야영장', 'BROWN-DOT호텔', '하얀미소',
                   '사마솔숲',
                   '햄릿과올리브', '클레시움아파트', '아파트', '이승만 김일성별장', '별침대', '벡스코역', '캠핑장',
                  '제트시티', '대명르미엘', '삼덕빌딩', '구미역', '어쩌다어달', '봉산극기체험센터', '에어 비앤비', '온천동',
                   '이안테라디움광안오피텔', '자무인텔', '신짬', '울산대학교/해송홀', '대구메디스퀘어', '뉴영보벤처타운',
                   '대우마리나1차아파트', '청주시외버스터미널', '광안에이파크오션오피스텔', '월소택지길', '문막교',
                   '청일관광농원', '라인2차빌라', '공현진1리해수욕장', '공현진1리해변야영장', '월정교 공영주차장',
                   '황리단길', '두계은농재', '오대산 월정사자연명상마을', '오대산자연명상마을', '양남항구길73',
                   '가비펠리치광안', '부선광역시 수영구 광남로100', '김해가야테마파크철기체험장', '제부도',
                   '산본시장고객지원센타', '독산성및세마대지(사적제140호)', '숭산초등학교', '사근진해변 주차장',
                   '정독도서관', '동호해변 ( 개인카라반 이용 )', '칠봉체육공원 인근', '한일오르듀', '노마즈하우스',
                   '최남단마라도가는여객선매표소', '우석대학교신학협력관', '이안테라디움광안오피스텔',
                   'CU 안동동문점', '해운대구중동마린타워', '강남하이엔드오피스텔', '강남하이엔드',
                   '타워더모스트광안오피스텔', '광안리 투헤븐', '캔버라타운', '사천해변', '큐브빌딩', '삼환미포씨랜드',
                   '월포여름치안센터', '노블레스', '삼산밀면전문점', '웅지세무대학교', '태평양임대아파트',
                  '제주출발 부산행 크루즈 배 선박 안 2등실 6호(6인실) 배정되어 숙박',
                   '부산출발 제주행 크루즈 배 선박 안 1등실 210호(4인실) 배정되어 숙박','정직한돈 협재직영점',
                   '평대바조', '용현현대아파트', '효원공원관리사무소', '힐스테이트광교아파트', '광교중흥S클래스아파트',
                    '가시버시', '자상한시간', '남송이오름', '장골해변', '소래포구종합어시장', '건봉빌딩', '거제시농업개발원',
                   '머묾한동', '세화해변','제주살이', '외돌개 일원', '한밭수목원', '애플하우스2','최남단 해안로',
                   '전주고속버스터미널', '송광사', '제주난타파크', '안성팜랜드','함양남계서원(사적제499호)',
                   '하고수동해수욕장버스정류장', '강릉역', '죽도해수욕장공중화장실', '신동해녀탈의장', '사일리커피 옆 공터',
                    '성산리마을회관', '서귀포예술의전당대극장', '한림공원', '롯데마트제주점', '도노미스테이', '강화참성단',
                   '라온더마파크','새별오름 주차장',
                    'NH중문농협중부지점','자가캠핑카(예래동)','화성행궁','서귀포항','썬다이브', '용두암1차현대아파트',
                   '제주지방경찰청한라경찰수련원',
                    '효돈동주민센터', '인천 강화군 삼산면 삼산서로 39-75', '오목대', '백강수피아빌리지3차',
                   '백강수피아빌리지 3차', '배론성지(충청북도기념물제118호)',
                    '경복궁태원전','월미도테마파크', '퀸제누비아호', '상천역', '섬머그리스게스트하우스', '섬스튜디오',
                   '부산항연안여객터미널',
                    '아부오름', '담은스테이', '내수전몽돌해변', '김녕미로공원', 'breezy studio', '교래흑돼지 본점',
                   '제이투패밀리호텔', '인피니티리조트A동',
                    '신경주역', '동광해운(주)', '동남식당', '임당동성당', '목포항 국제여객터미널', '마코다이브',
                   '금안한글체험휴양마을', '예송리해수욕장', '군산공항',
                    '서빈백사', '아진아트빌', '스테이사띠', '제주시', '독채팬션','제주야구인의마을', '성동시장공영주차장',
                    '두모어촌계복지회관', '제주 제주시 애월읍 애월해안로 454-10', '야영장', '비앤비', '제주온도',
                   '신제주연동트리플시티아파트',
                    '인천연안여객터널(비욘드트러스트 크루즈)', '울릉도산채영농조합', '영일만 신항 승객무료 주차장',
                   '제주참??가마', '마레1440', '저동항여객선터미널'
                    , '시즌빌딩', '한국지능정보화진흥원NIA글로벌센터', '중앙동행정복지센터', '조이힐하우스 자연치유센터',
                   '센트럴프라자', '비욘드트러스트호(인천-제주행)',
                    '목포연안여객선터미널', '포항지방해양항만청창고', '오케이호텔', '제주서귀포혁신도시LH1단지', '코하쿠',
                   '리스토리', '민박', 'GS25 제주함덕점',
                    'THECLOUDHOTEL', '금돈지', '성산포수협 활어위판센터', '올레렌트카', '여객선터미널', '크림빌',
                   '와온어촌계건물', '정밀교',
                    '풀덤 전북대점', '국립생태원방문자센터', '국립생태원 교육생활관', '마곡사성보박물관','포천아트밸리',
                   '대전', '신천지아파트',
                    '현대오일뱅크전주현대주유소','연화장', '약수동산', '정한파인시티', '정이품송관광농원',
                   '웅진스타클래스도램마을1단지/경비실',
                    '카라호텔','하이스트빌', '와온해변', '대승빌딩', '갤러리모란', '홍보성', '나주영상테마파크',
                   '중부대학교/청강관',
                    '오송산단관리사무소', '소선암자연휴양림', '금강자연휴양림', '조류마을', '지방행정연수원',
                   '오천농협하나로마트 원산도지점',
                    '목도강', '고창갯벌오토캠핑,글램핑장관리사무소','천둥관광지물놀이장','동서울종합터미널',
                   '국립중앙과학관', '합천영상테마파크관리사무소',
                    '서부버스터미널', '남원초원교회', '현대서적', '구미역', '노났네송가네', '부여군청소년수련원',
                   '광주 동구 금남로 202', '충북 충주시 신니면 내포길 37',
                    '구드래나루터','천은사(전라남도문화재자료제35호)', 'BLBETTERLIFE&COMPANY', 'MI무인호텔',
                   '저두출렁다리', '성보전', '문암2리어촌계',
                    '펜타폴리스II','한려해상국립공원','조령산자연휴양림백두대간생태교육장','프라지움11차',
                   '드림시티드림시티', '지우당', '강문해수욕장',
                    '행신역','친척 별장','한국철도공사대전충남본부','오대산자연명상마을','신두리해수욕장',
                   '벌천포해수욕장','무주구천동관광특구 주차장',
                    '천지가','신륵사','금돈지','전주풍패지관(보물제583호)','송도아트포레푸르지오시티오피스텔',
                   '라온수상레져타운','아람빌딩',"Who's next",
                    '시골마당','NH-OIL서부농협주유소','AENTRANCE','충청북도 영동군 상촌면 물한리 산 1-1',
                   '전북 전주시 완산구 전라감영로 52 (에어비엔비로 예약)',
                    '에스디빌딩','당진시로컬푸드행복장터','별서담','그림속여유','캠핑 노지','경남아파트2단지',
                   '소백산국립공원초암탐방지원센타','송산리고분군모형관매표소',
                    '충남 공주시 계룡면 기산리 46번지','초원가든','무위사나한전','고택전통체험관학인당',
                   '풍덕주택 1길 33','카페 다온', '서봉빌딩',
                    '하이플러스오피스텔','지인 별장','태안둘레길캠핑장관리사무소','대전천','고려아파트',
                   '남흥경로당','요나성당','안솔기쉼터','조계산선암사대각암',
                    '내소사고려동종(보물제277호)','친지','천은사(전라남도문화재자료제35호)','노근리',
                   '국립신시도자연휴양림', '개도청석포','익산역'
                    '순천기관차승무사업소', '벽골제마을(주)농특산물홍보체험관','동양식품',
                  '우도', '뷰티풀스테이인토평','1965올레시장54번가', '제주살레공장', '삼천포신항여객터미널',
                   '제주곶자왈도립공원탐방안내소'
                   , '에코랜드테마파크', '호주도', '삼다사우나', '제주신화월드테마파크', '함덕빵명장',
                   '서귀포자연휴양림관리사무소',
                   '표선해수욕장 주차장', '르샤또드마메르', '제주농원', '제주국제공항', '다이나믹메이즈 제주도성읍점',
                   '디엔에이빌리지',
                   '제주도','삼양동해녀탈의장','국립공주대학교천안캠퍼스/제8공학관','동문시장고객지원센터',
                   '중문골프클럽', '김녕휴게소',
                   '중문관광단지안내소','화순항','공회당','동광해운(주)','국제공무원교육원버스정류장',
                   '협재회관','숙성도 중문점', 'M휴리조트2',
                   '평대리해수욕장','우림랜드','줌타워','김해국제공항','남원포구','제주안달루시아',
                   '순천종합버스터미널','하젠타운',
                   '미영이네식당','어느틈에벌써','바다애','제주참??가마','청주국제공항','0626stay',
                   '위미타임', '블루파크시티',
                   '고사리식당','와치하우스', '해밀렌트하우스', '공간인화','산방산','노형수퍼마?R', '노형수퍼마켙',
                   '연동리슈빌디에스',
                   '창경빌리지','마이생튜에리','TBN제주교통방송','힐라체애월홈타운', '신도2리회관', '돈걱정', '알음알음하도',
                    '펜션','보야지','소랑이싯다', '차박', '화성식당',
                   '제주출발 부산행 크루즈 배 선박 안 2등실 6호(6인실) 배정되어 숙박',
                   '부산출발 제주행 크루즈 배 선박 안 1등실 210호(4인실) 배정되어 숙박', 'ICC제주국제컨벤션센터', '슬로우어스',
                   '남산케이블카', 'KTX광명역', '춘천역사', '어촌체험안내센타', '하도해변', '동방홍', '창룡문',
                   '금서루', '핀크스골프클럽',
                   '독자봉','서귀포시공영주차장', '광안지웰에스테이트','충의정','항파두리항몽유적지',
                   '성산포항종합여객터미널주차타워',
                    '북수구광장', '정방폭포', '비양도', '하귀1리정자', '관음사등산로입구', '스카이워터쇼',
                   '샤워실', '천지연폭포', '전북고속정비공장', '안산버스터미널', '서편제촬영지', '하르방',
                   '대구민자역사', '이월드동물농장',
                   '난지천공원', '서귀포시공영주차장', '동문시장', '베드로수산물센터', '김해수로왕릉송화문',
                   '후포해변', '서울역버스환승센터',
                   '춘천역사','왕십리역', '한라수목원', '붉은오름자연휴양림', '교래자연휴양림', '하이하이볼',
                   '오설록티뮤지엄', '울릉크루즈선착장',
                   '세화해수욕장', '시흥갯골생태공원', '이호테우해수욕장', '대전정보문화산업진흥원', '올림픽공원','부영아파트',
                   '디앤디파트먼트 제주', 'D&DEPARTMENTJEJUBYARARIO', '광주송정역', '상효원식물원', '더꽃돈 협재점', '한반도여',
                   '여수항연안여객선터미널 주차장', '평화광장', '연희피아노음악교습소', '사계해변', '용마로1-2',
                   '노지캠핑', '무송베스트빌',
                   '화순금모래해변클린화장실', '성산포항종합여객터미널주차타워', '태성빌라', '표선해수욕장', '고덕그라시움아파트',
                   '견월교', '강정마을 생태축제', '강정마을 생태축제 행사장', '학포항', '산방산온천', '휘닉스제주섭지코지 주차장2',
                   '제주돌문화공원', '천제연폭포매표소', '부산역6번출입구', '스시도모다찌 제주점', '민속마을사거리', '논짓물해변',
                   '싱계물공원', '양양국제공항', '노숙인현장상담실', '약천사종무소','자가캠핑카(예래동)','여행끝',
                   '수종사', '흑돼지짜투리구이',
                   '성산일출봉', '제3고객주차장', '목포항국제여객터미널', '어영공원', '가평역', '서부시외버스터미널',
                   '충혼묘지', '성산포항종합여객터미널주차타워',
                   '제주평화렌트카','태흥1리쉼터','제주오석심공예명장관', '구포역', '중동프라자',
                  'awesome 게스트하우스', 'CITY호텔', 'D&DEPARTMENT JEJU d room','GS칼텍스삼성셀프주유소',
                  '파인뉴에비뉴빌딩', 'H호스텔', 'H호텔','Joy houe','KT수련관', 'Kt수련관','SAT대학입시학원',
                  '강화해상공원', '경남빌딩', '경상남도교육청 종합복지관','골든프라자','공영주차장, 공원',
                  '곽지해수욕장 주차장', '관성솔밭해변', '광치기해변 주차장', '광화문열린광장','교대타운',
                  '구례5일장', '구산해수욕장', '구시포해수욕장', '국민은행자산관리프라자', '국제회관주차빌딩',
                  '굴업도해변', '굿모닝타워', '그랜드빌라', '근린생활시설입구', '금역당사당및종가(경상북도유형문화재제25호)',
                  '김포아트빌리지아트센터','남열해돋이해수욕장','내소사고려동종(보물제277호)',
                  '내소사', '노형수퍼마?R','농협쉼터나동', '다도해식당', '단독주택(다가구)',
                  '담양김선기가옥(문화재자료제180호)', '대둔산도립공원주차장관리사무소','대성식당',
                   '덕동해수욕장','도시고양이생존연구소', '도씨에빛I빌딩', '도재명차','독산성및세마대지(사적제140호)',
                   '돌산마트', '동광해운(주)', '동야루별장', '두리농원(전남친환경농업교육원)',
                   '라온빌리지', '로머스파크오피스텔', '롯데리아부산민락점', '롯데몰송도캐슬파크','리젠시모텔',
                   '마산고속버스터미널', '마포sk허브블루오피스텔', '망상한옥타운', '면암사', '명동센트럴빌딩',
                   '명산동 가옥', '목포지방해양항만청 국제여객터미널','무등산생태탐방원탐방교육관',
                   '문산빌딩', '미도주택', '바누카페', '배 (목포->제주)', '밧개해수욕장', '배론성지(충청북도기념물제118호)',
                   '백두대간 생태수목원 산림문화체험단지 체험센터', '베이징1','벽골제마을(주)농특산물홍보체험관',
                   '벽초지문화수목원','보리암보광전', '봄마중영농조합법인', '봉산동1369업무시설(주식회사세명씨엔씨)',
                   '부산은행남포동지점', '부산출발 제주행 크루즈 배 선박 안 1등실 210호(4인실) 배정되어 숙박',
                   '북부시외버스터미널', '비욘드트러스트호(인천-제주행)', '사라 스튜디오', '산장식당휴게소',
                   '삼척해수욕장', '새군산식당', '새서울빌','서귀포우체국', '서귀포칠십리시공원','서산프라자',
                   '서초휴양소', '석모도 만남의광장', '선유도해수욕장', '선물나라', '설악동 주차장', '성게돌',
                   '섬고래', '성산유치원', '성산포항종합여객터미널', '세종포레뷰1차', '세종한신휴시티',
                   '센트럴파크오피스텔', '셋,별','회사 복지시설', '회사 복지 시설',
                   '화현면 우시동길 239', '화진포해양박물관 주차장', '화양교','한국해양과학기술원 울릉도독도해양연구기지',
                   '폴바셋 영종도파라다이스시티점','통영시 중앙로175','탄천휴게소 논산방향', '천수만 A지구 상류', '천수만사기리수로', '천백고지휴게소',
                   '주차장', '조일프라자골프클럽', '제주행 여객선 비욘드트러스트 호','제주출발 부산행 크루즈 배 선박 안 2등실 6호(6인실) 배정되어 숙박',
                   '전진2리마을복지회관', '전주', '전북 전주시 완산구 전라감영로 52 (에어비엔비로 예약)','전라남도농업기술원과수연구소완도시험지',
                   '장호해변', '인천해양경비안전서하늘바다해양경비안전센터', '인천연안여객터널(비욘드트러스트 크루즈)', '인천문화예술회관대공연장',
                   '익산역', '유성푸르나임/경로당', '유관순열사생가지(사적제30호)', '원조태평소국밥', '웅천해변친수공원해양레저스포츠체험샤워장',
                   '웅천롯데캐슬마리나오피스텔', '울산대학교/청운학사기린관', '완도항 제3부두', '오죽헌', '예당무지개좌대', '연곡해수욕장관리실',
                   '연곡사', '여수시종합터미널', '엠에스페리(뉴스타호 배)', '에어비엔비 라누이블랑슈', '에어비엔비', '양촌공판장', '야영지', '신흥덕롯데캐슬레이시티/경비실2',
                   '신도시셀프세차타운', '슈퍼체인롯데슈퍼상품공급점', '송공산분재공원',
                  '회사 복지시설', '회사 복지 시설', '화현면 우시동길 239', '화진포해양박물관 주차장', '화양교','한국해양과학기술원 울릉도독도해양연구기지',
                    '폴바셋 영종도파라다이스시티점','통영시 중앙로175','탄천휴게소 논산방향', '천수만 A지구 상류', '천수만사기리수로', '천백고지휴게소',
                    '주차장', '조일프라자골프클럽', '제주행 여객선 비욘드트러스트 호','제주출발 부산행 크루즈 배 선박 안 2등실 6호(6인실) 배정되어 숙박',
                    '전진2리마을복지회관', '전주', '전북 전주시 완산구 전라감영로 52 (에어비엔비로 예약)','전라남도농업기술원과수연구소완도시험지',
                    '장호해변', '인천해양경비안전서하늘바다해양경비안전센터', '인천연안여객터널(비욘드트러스트 크루즈)', '인천문화예술회관대공연장',
                    '익산역', '유성푸르나임/경로당', '유관순열사생가지(사적제30호)', '원조태평소국밥', '웅천해변친수공원해양레저스포츠체험샤워장',
                    '웅천롯데캐슬마리나오피스텔', '울산대학교/청운학사기린관', '완도항 제3부두', '오죽헌', '예당무지개좌대', '연곡해수욕장관리실',
                    '연곡사', '여수시종합터미널', '엠에스페리(뉴스타호 배)', '에어비엔비 라누이블랑슈', '에어비엔비', '양촌공판장', '야영지', '신흥덕롯데캐슬레이시티/경비실2',
                    '신도시셀프세차타운', '슈퍼체인롯데슈퍼상품공급점', '송공산분재공원', '금역당사당및종가(경상북도유형문화재제25호)',
                    '소소한','YOLOHA하우스커피','룩앳더바리스타 송정점','지리산생태체험관','태종로716-4', '태종로 718-4', '이천쌀밥나랏님본관',
                    '독산성및세마대지(사적제140호)', '동호해변 ( 개인카라반 이용 )', '동호해변 ( 개인 카라반 이용 )','청송사','문경새재자연생태전시관',
                    '수라간','한빛길 28번길 34-1','담양김선기가옥(문화재자료제180호)', '초계국수', '무박', '아침가리계곡','주문진라일플로리스4차', '라이플로리스',
                    '봉산동1369업무시설(주식회사세명씨엔씨)', '단독주택', '청룡재', '열명의 농부', '마을회관(가리점마을쉼터)', '복래장', '무주반딧불장터',
                    '송공항', '구읍', '태권도원', '사정공원', '공기리', '곰소항', '대흥사', '벽골제마을(주)농특산물홍보체험관', '순천기관차승무사업소',
                    '여수당', '충청북도 영동군 산촌면 물한리 산 1-1', '휴가', '남부시장', '요가든', '명인사우나', '칸티빌음악학원', '갤러리노리', '중앙대후문',
                    '세봄빌', '동명동 좋은날', '승주컨트리클럽', '회사사택', '비욘드트러스트호(인천-제주행)', '삼무뚝배기갈치', '배 (목포->제주)', '갑문다리',
                    '삼광빌딩', '동광해운(주)', '배론성지(충청북도기념물제118호)', '정원책방', '부산출발 제주행 크루즈 배 선박 안 1등실 210호(4인실) 배정되어 숙박',
                    '노형수퍼마켙', '현대썬앤빌오피스텔', '을왕4통마을회관', '밭지름해수욕장', '개머리언덕', '한탄강관광지관리사무소',
                   '작은풀안해변', '반포식스', '트윈시티남산오피스텔', '함박골마을영농조합법인','하모니빌딩', '삼둔사가리',
                   '은하빌라', '웅천친수공원', '충남 탸안군 안면읍 남면 곰섬 활토빌리지', '충청남도 천안시 동남구 유량동 210-6',
                   '서동신부동산', '공주시사곡면사무소', '개버랜드', '한강시민공원여의도지구', '하나로마트', '수원화성박물관']

# 지정된 단어가 포함된 행을 제거함
dff = dff[~dff['VISIT_AREA_NM'].str.contains('|'.join(words_to_remove))]

# 의미 없는 변수 추가 정의
words_to_remove1 = ['부산출발 제주행 크루즈 배 선박 안 1등실 210호(4인실) 배정되어 숙박', '제주출발 부산행 크루즈 배 선박 안 2등실 6호(6인실) 배정되어 숙박',
                   '인천연안여객터널(비욘드트러스트 크루즈)', '유관순열사생가지(사적제30호)', '엠에스페리(뉴스타호 배)',
                   '금역당사당및종가(경상북도유형문화재제25호)', '동호해변 ( 개인카라반 이용 )', '동호해변 ( 개인 카라반 이용 )',
                   '독산성및세마대지(사적제140호)','담양김선기가옥(문화재자료제180호)','봉산동1369업무시설(주식회사세명씨엔씨)',
                   '마을회관(가리점마을쉼터)', '벽골제마을(주)농특산물홍보체험관', '비욘드트러스트호(인천-제주행)','노형수퍼마켙',
                   '배 (목포->제주)', '동광해운(주)', '배론성지(충청북도기념물제118호)',
                    '부산출발 제주행 크루즈 배 선박 안 1등실 210호(4인실) 배정되어 숙박','??에버']
len(words_to_remove1)

# 기존 방식에서 제거 되지 않은 변수 추가 제거
for i in range(len(words_to_remove1)):
    dff[dff['VISIT_AREA_NM']==words_to_remove1[i]].index
    dff=dff.drop(dff[dff['VISIT_AREA_NM']==words_to_remove1[i]].index)

dff=dff.drop(dff[dff['VISIT_AREA_NM'].str.contains('라마다앙코르해운대호텔라')].index)

fdf = dff.copy()

# 주소지는 다른데 같은 이름 혹은 주소지는 같은데 다른 이름 등 변수 통일화
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('테스모텔', '넘버25 수원화성행궁')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포세이돈수상레저', '청평포세이돈카라반')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에버랜드스피드웨이', '에버랜드 홈브리지캐빈호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬스타모텔', '썬스타호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬스타', '썬스타호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('만월문(Mamwolmoon)', '그랜드인터컨티넨탈 서울파르나스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수송타워', '신라스테이 광화문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연채천', '메이필드호텔 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반딜라이트', 'AC호텔 바이 메리어트 서울강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('IFC센터', '콘래드 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해슬리나인브릿지골프장', '해슬리나인브릿지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해슬리나인브릿지CC', '해슬리나인브릿지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국민은행태평로지', '코리아나호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비스타워커힐서울명월관', '그랜드워커힐서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽라운지앤풀바', '파라다이스시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('효성해링턴스퀘어', '글래드호텔 마포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주식회사인디자인25', '남산포레스트게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('충무빌딩', '더캡슐 명동점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴호텔 평창 체크인', '켄싱턴호텔 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연인산유황온천리조트', '연인산리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('AMANTIHOTELSEOUL', '아만티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PINECITYHOTEL', '파인시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('판교미래에셋센터', '반포식스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠블호텔킨텍스세탁실', '엠블호텔킨텍스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Ruhe', '루헤풀빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인천에어포트호텔', '베스트웨스턴프리미어 인천에어포트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이클래스', '하이클래스펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리틀미코노스 선재도', '리틀미코노스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('곤지암리조트주차장', '곤지암리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포스코타워송도', '오크우드프리미어 인천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠블호텔킨텍스세탁실', '소노캄 고양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한국도로공사', '나인트리프리미어호텔 서울판교')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PARADISECITY', '파라다이스시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경포더블루핀신축공사(주)홍성건설', '하이오션 경포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캡텍', '엠앤럭키호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립유명산자연휴양림 산림문화휴양관', '국립유명산자연휴양림수련관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('중미산천문대', '중미산천문대숙소')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('J잔디가든유원지', '캠프99')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('미조사', '호텔케이월드청계')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JCT.WORLDGATE', '영종스카이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코리아.삼성.클로버주차', '영종스카이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인천환경공단송도사업소', '인천환경공단송도스포츠파크캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마이다스리조트', '마이다스호텔&리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드워커힐서울 더글라스하우스', '더글라스하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드팰리스호텔 인천', '그랜드팰리스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('컨싱턴호텔여의도', '켄싱턴호텔 여의도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('2030모텔입구', '브라운도트 인천주안역점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELSKYPARK', '호텔스카이파크 인천송도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아나의섬스파펜션', '대부도 로그인 펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('가평 숲으로가는길 펜션', '숲으로가는길펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('부림빌딩', '더리프 서울사당')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('킨텍스', '킨텍스바이케이트리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('밀레니엄서울힐튼주차장', '밀레니엄 서울힐튼')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스틴조선 서울지점', '웨스턴조선호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더 플라자 라운지', '더플라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드앰배서더서울', '앰배서더 서울풀만호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베스트웨스턴프리미어구로호텔', '포포인츠바이쉐라톤 서울구로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대부도펜션시티관리사무소', '대부도펜션시티')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연화도감', '허브펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제2롯데월드', '시그니엘 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데호텔월드', '롯데호텔 월드점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데호텔월드 라세느', '롯데호텔 월드점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('창진프라자', '호텔오로이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('밀리오레', '밀리오레호텔 명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스위스산장레스토랑', '스위스랜드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('광명역세권복합1주상복합태영데시강신축공사', '테이크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레스트레스토랑', '레스트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('달빛매점', '달빛정원글램핑앤캠핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('안다즈', '안다즈 서울강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GLADHOTELS', '글래드 여의도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼풍상가', '호텔피제이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('RGBSKY호텔', '에어스카이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('THEDESIGNERS', '호텔더디자이너스 DDP점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('정휘빌딩', '소테츠프레사인 서울명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메리어트 이그제큐티브아파트먼트 서울, 수영장', '메리어트이그제큐티브아파트먼트 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('영종오션솔레뷰', '오션파크나인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울백팩커스', '머스트스테이호텔 명동점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국도호텔', '호텔국도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오하브', '오하브글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수목원프로방스', '수목원프로방스캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서원빌딩', '스텔라호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Monterey17Hotel', '몬터레이17호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SK명동빌딩', '이비스앰배서더 서울명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('선녀바위', '을왕4통마을회관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GOLDENTULIP', '골든튤립에버용인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('네스트호텔 사우나', '네스트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('네스트호텔 피트니스', '네스트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서머셋빌딩', '서머셋팰리스서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('도쿄인', '토요코인호텔 영등포점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('WESTRAVENHOTEL', '웨스트레이븐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수원 No.25-화성행궁점', '넘버25 수원화성행궁')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라이즈오토그래프컬렉션', '라이즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELTHEMAY', '호텔더메이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청산농원', '포시즌스오토캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('을왕리해수욕장 (삼양민박)', '삼양민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SKY모텔', '브라운도트호텔 종로점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마실캠프', '마실빌리지펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('길조', '길조호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인부산지점', '토요코인호텔 부산서면')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스엠배서티호텔', '이비스앰배서더호텔 인사동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('진성빌딩', '더드림호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용인시 처인구 김령장동 SR디자인 호텔', 'SRR호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스탠포드호텔', '스탠포드호텔 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이동식 캠핑카라반', '이동식캠핑카라반')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PATIO7', '파티오세븐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레드썬모텔주차장', '레드썬모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('THEPRINCEHOTEL', '더프린스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SEA앤MOONTOURST호텔', '해월관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('천호 August', '호텔8월 천호')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔드마르', '제부도케렌시아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('세종호텔주차타워', '세종호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SOLBEACH진도', '쏠비치 진도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('허가96152', '호텔더디자이너스 종로점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파라다이스시티 레드윙', '파라다이스시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파라다이스시티', '파라다이스시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인천 파라다이스 호텔', '파라다이스시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이호테우해수욕장', '어나더153')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('산방산온천', '레스케이프호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청산수목원', '미스터오션펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서머셋팰리스서울', '서머셋팰리스 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('케이팝하우스', '케이게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('단골민박', '콘래드 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수세계박람회주제관해양베스트관', '그레이캐슬펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이순신광장', '그레이캐슬펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('만민장', '모텔요기')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경산빌딩', '호텔베뉴지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('두바스모텔', 'MY24모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('광암라이피아빌딩', '명동멀린호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('김해수로왕릉송화문', '더휴식 이로호텔 월곶지점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쉘부르', '더휴식 이로호텔 월곶지점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('TOPCLOUP호텔', '탑클라우드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다앙코르바이윈덤 김포한강호텔', '라마라앙코르김포한강호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데호텔', '롯데호텔 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('융릉건릉역사문화관', '라마라앙코르김포한강호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('왕십리역', '인천스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코텔', '체스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오송역', '호텔푸르미르')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서귀포시공영주차장', '서울숲스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무의스토리', '해둥실펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('째즈하우스모텔', '케이팝호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SHILLASTAY', '신라스테이 서대문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HT오피스텔', '신라스테이 구로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블룸비스타호텔', '양평블룸비스타호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬샤인신호텔별관', '썬샤인신호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨사이드에비뉴', '오션파크나인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('강원대학교', '호텔세븐스텝 시흥정왕점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해돋이공원', '포포인츠 바이 쉐라톤 조선 서울명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용담공원', '호수산장캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELYAJA', '호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('럼스하우스모텔', '월하여관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한강빌딩', '24게스트하우스 잠실점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('광안리해수욕장', '강화도 더스타스파펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그레이스리호텔', '호텔그레이스리 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('화순금모래해수욕장', '갯벌놀이펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더스테이트호텔', '더스테이트 선유')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('VISTACAY호텔', '비스타케이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('도들봉공원', '또바기애견글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금진해수욕장', '이비스엠배서더호텔 서울강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('구포역', 'G7호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용문산관광단지', '용문산야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이호테우해수욕장', '마이다스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대한제분', '로이넷호텔 서울마포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다송도', '송도라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('자연체험학교', '상아골계곡오토캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('G7', 'G7호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('세빛둥둥섬', '그랜드하얏트서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스틴조선서울 라운지', '웨스틴조선서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스틴서울 아트갤러리', '웨스틴조선서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신세계백화점', 'JW 메리어트 호텔 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('허브하우스', '호텔베뉴지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('은가비펜션', '글램독펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('당진1차푸르지오아파트/상가', '노보텔앰배서더 서울강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서문시장', '글래드 여의도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔스퀘어', '호텔스퀘어 안산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SEASONEPISODE', '에피소드706')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELKWORLD', '호텔케이월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('루시루시', '루시호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아하청소년성문화센터', '하이서울유스호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봉황모텔', 'R게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더디자이더스호텔', '호텔더디자이너스 동대문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('송도아트윈푸르지오아파트', '홀리데이인인천송도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반스테이 인천송도', '어반스테이 송도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라비돌컨트리클럽', '라비돌호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이병택정형외과', '훼미리사우나파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PATIO7', '파티오세븐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PREMIEHOTELXYM', '프리미어호텔XYM')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아카데미타워', '라성보석사우나')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PARKMARINEHOTEL', '파크마린호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('별천지', '예담계곡펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELYAJA', '호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수모텔', '넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼영빌딩', '크라운파크호텔서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웅진타워', '썬사우나')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('FOURPOINTS', '포포인츠바이쉐라톤 서울강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('뉴엘지프라자', '스파렉스 불가마사우나')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트청평', '켄싱턴리조트 가평')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HAVENUE광광호텔', 'HAVENUE관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('애경타워', '홀리데이인익스프레스 서울홍대')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용문사대웅전', '용문사양평 템플스테이')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '울산 남구 삼산로 308':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('토요코인호텔', '토요코인호텔 울산삼산점')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '토요코인호텔':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('토요코인호텔', '토요코인호텔 인천부평점')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('성지빌딩', '야코리아호스텔 강남점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('센트럴파크호텔', '송도센트럴파크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('정도트윈빌', '스타즈호텔 독산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울드레콘시티', '서울드래곤시티')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소피텔 앰배서더 서울 온천/사우나 16층', '소피텔 앰배서더 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('su모텔', '넘버25 방이점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아보헌', '아보헌 전통숙소')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('NINE9HOTEL', '나인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('문경 라마다호텔', '라마다호텔 문경')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('속초라마다설악해양호텔', '라마다호텔 속초')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '서울 중구 동호로 354':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔', '라마다호텔 서울동대문')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '대전 유성구 계룡로 127':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔', '라마다호텔 대전')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 서귀포시 김정문화로27번길 9-1':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔', '스위트오션 스위트메이서귀포호텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 서귀포시 천제연로 153':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔', '해리안호텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '서울 구로구 경인로 624':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔', '라마다서울신도림호텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '인천 남동구 소래역로 36':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔', '라마다인천호텔')
fdf = fdf[fdf['VISIT_AREA_NM'] != '라마다호텔']
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다호텔평창', '라마다호텔 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인부산역Ⅰ', '토요코인호텔 부산역1호점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽이에스통영리조트', '클럽ES리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비발디파크컨트리클럽', '소노벨비발디파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨비발디파크 B동', '소노벨비발디파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴호텔', '켄싱턴호텔 설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('B호텔', '브라운도트호텔 신호점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이캐슬콘도', '하이캐슬리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('올림피아사우나', '올림피아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에스피에스타', '에스피에스타펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('중도연수원건립사업공사', '황룡원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주힐튼호텔선재현대미술관', '힐튼호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동행', '경주동행스테이')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 정선군 고한읍 물한리길 8':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('정선리조트', '메이힐스리조트')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '정선리조트':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('정선리조트', '썬라이즈호텔 성산')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)파라다이스호텔부산', '파라다이스호텔 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑스스크린골프', '탑스텐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알펜시아컨트리클럽', '알펜시아 홀리데이인리조트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주현대호텔', '라한셀렉트 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알펜시아 홀리데이인&스위트콘도', '알펜시아 홀리데이인리조트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스테이예스', '꿈의궁전모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한마음모텔', '낙산한마음콘도텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블루원게임존', '블루원리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('POEM모텔', '포엠모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PARKROCHE', '파크로쉬리조트앤웰니스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스무디킹', '신라스테이 해운대')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한마음레프팅', '한마음펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KT경주수련관 구내식당', 'KT경주수련관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스카이베이경포호텔', '스카이베이 경포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PARKHYATTBUSAN', '파크하얏트 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캔싱턴리조트', '켄싱턴리조트 설악비치')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덕두원오리숯불구이', '로하스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타', '덴바스타호텔 헤리티지 경주점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('뉴랜드', '오게 부산역점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('평창MediaResidence', '평창미디어레지던스')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경남 통영시 항남4길 19':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('에쿠스모텔', '호텔피코 통영지점')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '부산 해운대구 해운대로 157':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('에쿠스모텔', '라찌모텔')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서부어항', '호텔레이지헤븐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다가구주택', '블루아라펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엘리시아빌딩', '엘리시아부티크호텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 속초시 청초호반로 291':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('대원마트', '썬라이즈호텔 속초')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '강원 홍천군 두촌면 광석로 898-87':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('대원마트', '세이지우드 홍천')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('골든비치골프리조트', '설해원')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 춘천시 중앙로27번길 11':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('그랜드모텔', '춘천시그랜드모텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '강원 홍천군 두촌면 광석로 898-87':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('그랜드모텔', '브라운도트 제천역점')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비발디파크오크동', '소노벨비발디파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비발디파크소노펠리체D', '소노벨비발디파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노펫클럽앤리조트 비발디파크', '소노벨비발디파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비발디파크오션월드', '소노벨비발디파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나동', '스테이캠프 펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오시리아스위첸마티에', '마티에 오시리아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주식회사씨에이치에이퍼시픽', '썬클라우드호텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경북 울진군 북면 덕구온천로 921':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('GS25', '덕구온천리조트')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '강원 양양군 현북면 하조대해안길 11':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('GS25', '알프스비치')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '경남 거제시 동부면 학동7길 2':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('GS25', '스터번호텔')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알프스비치콘텔', '알프스비치')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('벽산덕구온천콘도', '덕구온천리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새막골캠프', '새막골캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트별관', '한화리조트설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트본관', '한화리조트설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 설악별관', '한화리조트설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 설악쏘라노', '한화리조트설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포천한화리조트', '한화리조트 산정호수')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '부산 해운대구 마린시티3로 52':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('한화리조트', '한화리조트 해운대')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '강원 평창군 봉평면 태기로 174':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('한화리조트', '한화리조트 평창')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '강원 속초시 미시령로2983번길 16':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('한화리조트', '한화리조트설악')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스파크레드앤핑크콘도미니엄', '한화리조트 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('로얄호텔', '부곡로얄관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('앙코르호텔', '')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다앙코르호텔', '라마다앙코르 부산역호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('앙코르호텔', '라마다앙코르해운대호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더우', '대구메리어트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라발스호텔', '라발스스카이카페&바')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('페리아902', '페리아902 오션뷰스파펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('FORENATHEISLAND', '한화호텔앤드리조트 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원추추파크네이쳐빌', '하이원 추추파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('황남관한옥체험마을', '황남관한옥호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장호프라자', '삼척펜션오라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('ST.JOHNSHOTEL', '세인트존스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금호리조트(주)설악', '설악금호리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이지스', '이지스모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스파트리빌리조트', '더앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('구재봉자연휴양림휴양관', '구재봉자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더베네치아레지던스&호텔', '더베네치아스위트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('춘천언어치료센터', '봄스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아난티코브', '아난티 힐튼 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('워터하우스', '아난티 힐튼 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주식회사블루호라이즌', '더블루테라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('IBISBUDGET호텔', '이비스앰배서더 부산해운대')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노문 델피노 East동', '소노펠리체 델피노')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BEACHHILL', '비치힐풀빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('셔블', '행복한옥마을 셔블')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('MU모텔', '호텔라인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라LAKAISSANDPINE', '라카이샌드파인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('69모텔자갈치해수탕', '자갈치해수탕')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경남상회', '행복민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('풍뎅이호스텔 별관', '풍뎅이호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서핑스쿨', '서퍼랑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('N°25호텔', '넘버25호텔 포항대잠점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('낙산사로41 동신민박', '동신민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더잭슨에스테틱', '더잭슨나인스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비슬산자연휴양림관리사무소', '비슬산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('숲속수련장위생복합시설', '대관령자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('초콜릿', '베네시안모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('목련장여관', '블로바이블로호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경동리인타워', '리인스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비바체리조트', '비바체하동호리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('RIVEERTAIN호텔', '리버틴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대구메리어트호텔 어반키친', '대구메리어트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하늘바라기글램핑장관리사무소', '하늘바라기글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CNN타워', 'CNN호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('황룡stay', '황룡원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하비리조트주식회사', '하비카라반럭셔리글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('성건동2호태양광발전소', '포테이토모텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '울산 남구 삼산로 200':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('신라스테이', '신라스테이 울산')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '서울 동작구 시흥대로 596':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('신라스테이', '신라스테이 구로')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이호텔', '신라스테이 서초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이제주호텔', '신라스테이 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이주차타워', '신라스테이 천안')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아이파크콘도IPARK콘도', '아이파크콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('럭키센타빌딩', '호텔여기어때 묵호점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나봄리조트', '무주나봄리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽하우스', '탑스텐리조트 동강시스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서커스', '켄트호텔 광안리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경보장', '미드나잇인경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼성홈프레스티지I', '월드스카이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대명콘도미니엄', '델피노 리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('올림피아사우나', '올림피아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('TheK경주호텔스파월드', '더케이호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대명르미엘', '더마크속초레지던스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용화산자연휴양림관리사무소', '용화산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쏠비치휴양콘도미니엄', '쏠비치 양양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신축허가0314호', '미아모르모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('락토장', '더레드하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대명르미엘', '더마크속초레지던스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동성유료주차장', '골든튤립호텔 남강')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아점오브시카고', '강릉씨티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('농협경주교육원 휴양관 주차장', '농협경주교육원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에어포트콘도텔3관', '에어포트콘도텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아르반호텔BNK부산은행서면롯데1번가지점', '아르반호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에어포트콘도텔(4관)', '에어포트콘도텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('필로스호텔, 옥희커피', '필로스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SEASON모텔', '브라운도트호텔 강릉경포점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('미스터힐링', '라마다앙코르해운대호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다정선호텔', '라마다앙코르정선호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('THE묵다', '호텔묵다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('NOMAD호텔', '노마드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한창빌딩', '송모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('델피노골프&리조트', '소노펠리체 델피노')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경북 경주시 보문로 422':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('코모도호텔', '코모도호텔 경주')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '부산 중구 중구로 151':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('코모도호텔', '코모도호텔 부산')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인구다이브리조트', '브리드호텔 양양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SLOWOCEAN', '슬로우오션&히든포레스트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('ALMOND호텔', '아몬드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('산마을송원농장민박', '송원농장민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('송원산마늘농장', '송원농장민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOUNDHOTEL', '하운드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장유중원복합시설', '장유아몬드키즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한산마리나호텔&리조트', '통영한산마리나호텔리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('통영 한산마리나호텔&리조트', '통영한산마리나호텔리조트')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 원주시 지정면 신지정로 192':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('서정프라자', '브라운도트호텔 원주기업도시점')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '경남 진주시 에나로 103-9':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('브라운도트호텔', '브라운도트호텔 진주충무공점')
fdf = fdf[fdf['VISIT_AREA_NM'] != '브라운도트호텔']

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 순천점', '브라운도트호텔 순천역점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KT경주수련관  507호', 'KT경주수련관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('영동종합철강', '이지무인텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동양장', '동양모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('LAKEOCEAN', '레이크오션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('황매산미리내파크 제1주차장', '황매산미리내파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해피하우스(HAPPYHOUSE)', '해피하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유모텔', 'U모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주 J 부티크 호텔', '제이모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아리랑호텔A동', '아리랑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아리랑호텔B동', '아리랑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GOLDENBAY', 'GB골든베이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주식회사건국경주지점', '경주봉황맨션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주경빌딩', '휴플러스게스트하우스')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경남 김해시 번화1로 52':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('넘버25호텔', '장유넘버25호텔')

fdf = fdf[fdf['VISIT_AREA_NM'] != '넘버25호텔']

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 강릉시 창해로 439':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('넘버25', '경포수호텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '강원 춘천시 영서로 2515':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('넘버25', '수모텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '충북 보은군 보은읍 보청대로 1491':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('넘버25', '수호텔')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베스트루이스해밀턴Hotel', '베스트루이스해밀턴호텔 해운대점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더시크릿양양패밀리텔', '양양비치콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유원지', '루벤스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코레스코콘도슈퍼', '오션투유리조트 속초설악비치호텔앤콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원스키장', '하이원리조트 마운틴콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웰리힐리파크', '웰리힐리파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더케이호텔', '더케이호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('TheK경주호텔스파월드', '더케이호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴플로라호텔', '켄싱턴호텔 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('피어26호텔&파티', '피어26호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전주해장국', '꾸례게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장호프라자', '삼척펜션오라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('LightHousePoint', '라이트하우스포인트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스평창 그린동', '휘닉스 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이캐슬콘도', '하이캐슬리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주힐튼호텔선재현대미술관', '힐튼호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해운대구중동마린타워', '팝콘호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유일O.A시스템', '부산숙박닷컴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쏠비치양양 리조트동', '쏠비치 양양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELCUBE', '호텔큐브')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동해보양온천컨벤션호텔수영장', '동해보양온천컨벤션호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대암산용늪자연생태학교', '용늪자연생태학교펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쏠비치호텔앤리조트삼척', '쏠비치 삼척')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쏠비치', '쏠비치 삼척')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스평창 블루동', '휘닉스 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스카이베이 경포', '스카이베이경포호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('2004신축허가88호', '호텔해든')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('묵산미술박물관', '영월아트앤캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비치사이드모텔', '페블비치')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('시그니엘 부산 숙박객 라운지', '시그니엘 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('시그니엘 부산 인피니티풀', '시그니엘 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('시타디비스아파트호텔', '펠릭스바이에스티엑스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고래불하계휴양소', '고래불국민야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CIELOCEANHOTEL', '씨엘오션호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)평화지류', '브라운도트 호텔 서면범천점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('속초엑스포월드', '호텔웨이브')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('송정 제타-감동호텔', '감동호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('AVVIOHOTEL', '아비오호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('강동문화체험학교', '텐타우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한려해상국립공원', '클럽이에스통영리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽ES리조트', '클럽이에스통영리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립화성숲체원체험방', '미도주택')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하추자연휴양림휴게실', '하추자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('영월서부시장', '남이섬설리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('헤이춘천', '헤이춘천')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '부산 중구 중구로 151':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('코모도호텔', '코모도호텔 부산')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '경북 경주시 보문로 422':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('코모도호텔', '코모도호텔 경주')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코레일망상수련원', '국가철도공단 망상수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동심하리지구복합개발사업', '시타딘커넥트호텔하리 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주교육문화회관', '더케이호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아모르빌딩', '호텔딘')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PARKROCHE', '파크로쉬리조트앤웰니스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PARKROCHE', '파크로쉬리조트앤웰니스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('함양 “소소한”', '소소한')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('솔라피아호텔', '솔라리아니시테츠호텔 부산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('합천호관광농원', '합천호청소년수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한경직목사우거처', '켄싱턴호텔 설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아르반호텔BNK부산은행서면롯데1번가지점', '아르반호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼화빌딩', '바닷가펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('콩나물박물관', '호텔코지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오목대', '호텔코지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대한불교조계종내원정사', '내원정사 템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봉래초등학교문산분교', '성동힐링센터 휴영월캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('독일문화체험센터', '독일마을광장펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('황악산하야로비공원', '사명대사공원건강문화원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('STAY', '스테이호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동일당구장', '만월도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운하우스엔카페', '브라운스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('의암버스정류장', '개똥골펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('안동김씨태장재사', '이상루')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청사포정거장', '스카이베이 경포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스카이베이경포호텔', '스카이베이 경포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('모텔달러', '덴바스타호텔헤리티지 경주점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('골굴사종무소', '골굴사 템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BIZINN호텔', '호텔비즈인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('월화거리공영주차장유료', '쉼모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('가당빌딩', '경성여관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아스티호텔', '아스티호텔 부산역')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다비앙', '다비앙스파펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한라산', '호텔크라페 성서점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('지성요양병원', '그라시아스 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경포투썸', '메르뷰스파펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('락희원', '락희원게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주월드', '별헤는마을')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('죽북초등학교', '에코팜오토캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('단양역', '국립두타산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('몽브루모텔', '감동호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)삼성온천호텔', '삼성온천호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경보장', '미드나잇 인 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('빈폴타워', '강릉관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청해플라자', 'U모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('황실예식장', '신라부티크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('tt호텔', '티티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호수공원1주차장', '티티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('순천만생태공원', '글램독')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금강산모텔', '호텔리안')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('김녕미로공원', '삼성호텔 거제')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수해상케이블카놀아정류장', '스타즈호텔 울산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립아시아문화전당', '스타즈호텔 울산점')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '울산 남구 남중로 65':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('스타즈호텔', '스타즈호텔 울산점')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '서울 중구 수표로 16':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('스타즈호텔', '스타즈호텔 명동2호점')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용머리공원안내센터', '인문학마을캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SEACRUISEHOTEL', '씨크루즈호텔속초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유림정', 'SL호텔강릉')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데백화점', '롯데호텔 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('광릉', '마중펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대전복합터미널', '세인트존스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신호계타운', '세인트존스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('2Night호텔', '호텔투나잇')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한국해양대학교/국제대학관', '세르비호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('게스트하우스청춘', '청춘게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동성유료주차장', '골든튤립호텔남강')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('심미에셈빌', '퀸벨호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('농협창고', '패밀리하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해변원룸', '경성여관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('의기사', '까사드발리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수성빌', '리센오션파크속초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동서울종합터미널', '리센오션파크속초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대흥빌딩', '센텀프리미어호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대구연경대광로제비', '1984왕림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOVNDHOTEL', '베이하운드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무광빌딩', '블루밍호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한방산업단지힐링센터', '성주봉자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('THEK모텔', '더K모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션사이드', '미포오션사이드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('J호텔', '제이모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신대교휴게소', '브릿지호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동성유료주차장', '골든튤립호텔 남강')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('팔공산탐방지원센터', '애플호텔펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오리온사우나', '노블스테이호텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 태백시 석공길 25':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('궁전모텔', '블루문게스트하우스')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한성가든', '온도호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('숙박시설-퀸즈모텔', '퀸즈모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봉림빌딩', '창원M모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('바람주차장', '하이원그랜드호텔 컨벤션타워')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알프스별장', '호텔보보')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('지산고택(경북문화재자료제140호)', '하회마을 지산고택')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('밸류호텔강릉', 'SL호텔 강릉')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace("그랜드'엘시티레지", '시그니엘 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드엘시티레지', '시그니엘 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('OCEANTREE', '오션트리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('STAY251', '스테이251')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스아일랜드', '휘닉스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('가온제주', '가온제이스테이 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('뷰리조트비스타', '비스타뷰리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('디앤디랩', '아침해변펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELSIRIUS', '호탤시리우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('굿모닝하우스팬선', '굿모닝하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플레이스', '플레이스캠프 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('바솔트', '더그랜드섬오름 신관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레드스카이펜션B동', '노을을 담다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JAY&CHLOE', '제이앤클로이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주조천스위스마을', '스위스마을 컬러스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('PEPPERMINTLOUNGE', '누코지스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELTOPISLAND', '탑아일랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELWINSTORY', '윈스토리호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BRICKSHOTEL', '브릭스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해피데이', '호텔해피데이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('푸르다비치', '푸르다오션 펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('eq stay', '이큐스테이 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아르츠스파앤풀빌라안내', '아르츠스파앤풀빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SONOBELLE', '소노벨')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('DYIVEOCEANO', '다인오세아노 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('별하비', '별하비스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('온더게스트하우스 성산일출봉점', '온더스톤게스트하우스 성산일출봉점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아트스테이', '호텔브릿지 서귀포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('크럼스쿠키', '모노스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다인리조트E동', '다인리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('MAROONSTAY74', '머룬스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SunnyHOTEL', '써니호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아모렉스리조트본관A동', '아모렉스리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('팜빌리지', '비스비제주 앤 팜빌리지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주올레문', '제주올레로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주드루앙 북촌 프라방', '제주드루앙 신촌')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비욘드북카페앤게스트하우스 전기차충전소', '비욘드북카페앤게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('EASTERNHOTEL', '이스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주헬스케어타운휴양콘도미니엄2차', '제주헬스케어타운 리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션파크나인호텔', '오션솔레뷰호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 케니 기장', '호텔케니 기장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('COCONHOTEL', '코쿤호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BKHOTELJEJU', 'BK호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELMCC', '호텔MCC')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('G.HOTEL', '호텔 지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오누게스트하우스', '오누박스게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('감수굴', '감수굴 펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마라크라', '바라크라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('설레임게스트하우스 1호점', '설레임게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('드레비앙', '트레비앙')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비욘드북카페', '비욘드북카페앤게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('THEWEHOTEL', 'WE호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('루나인제주', '바구니민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('atcorner', '앳코너')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오후6시펜션', '오후여섯시펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('온더게스트하우스', '온더스톤게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔펄리PLUS', '호텔펄리플러스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SOLHOUSE', '솔루나하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔휘슬작', '호텔휘슬락')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주시외버스터미널', '하도36')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동홍2단지주공아파트', '홍스랜드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('두꺼비의빛', '뷰티풀메스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대왕암공원', '수선화민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제천역', '수선화민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('성이시돌목장', '호텔탑아일랜드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('명월상동회관', '명월다락')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이촌역', '명월다락')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('글로스터호텔', '글로스터호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('퍼스트70', '호텔노블리아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고수동굴', '호텔노블리아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('D&DEPARTMENTJEJUBYARARIO', '디앤디파트먼트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('랜딩관 제주신화월드호텔앤리조트', '제주신화월드호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔체크인', '체크인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한라산관음사일주문', '아루미호텔 협재점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동대문디자인플라자', '제주신화월드호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아트스테이함덕비치', '에반에셀호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더포그레이스호텔앤리조트', '더포그레이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신신호텔 천지연', '신신호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('지엔비어학원새서귀포캠퍼스', '하루하나')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('ICC제주국제컨벤션센터', '제주부영호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주레슬리', '레슬리호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덕수궁대한문', '서귀포시모구리야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주브릭스', '브릭스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('산방식당', '애월더선셋리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장안공원화장실', '켄싱턴리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('절물휴양림', '월령코지펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('월령코지', '월령코지펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('카페', '스테이예스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('춘천사이클경기장', '토다게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동복리671', '노을을 담다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('팔달문시장', '부영호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경기전사적지339호', '오리엔탈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('정발산공원', '패밀리아펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라메종베니인제주펜션', '라메종베니')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주허브동산', '더베스트제주성산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금서루', '호텔더그랑중문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주4.3평화교육센터', '제주에산다면')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('외돌개관광단지', '봄그리고가을리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('와이리조트 제주', 'Y리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('와이리조트', 'Y리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('늘작게하우스', '제주라일락')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아모렉스', '아모렉스리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포레나광교', '더포레스트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금서루', '올레브릿지하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고래불해수욕장', '호텔리젠트마린')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신성리조트C동', '씨오르리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해미읍성', '봉자게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메종글래드제주호텔', '메종글래드제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('견월교', '마레1440 펜션&카페')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔엘린', '엘린호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드밀리언스호텔 서귀포', '그랜드밀리언스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('예림원', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼무공원', '잇츠힐펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('잇츠힐', '잇츠힐펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔더본 제주', '호텔더본')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('갤럭시렌터카', '게으른소나기')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용연구름다리', '호텔서귀피안')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('보목포구', '플레이스 캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('조일리어촌계', '플레이스 캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('1100고지버스정류장', '숨게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('상하이', '카페호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('볼호텔', '그레이스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해뜸', '오션패밀리호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주엘루이호텔', '엘루이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('산토리', '아란치아 펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('둔산선사유적지', '펜션연리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주민속박물관', '바람에스치운다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메르블루게스트하우스', '바람에스치운다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('별방진', '씨오르리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유스퀘어광주종합버스', '아따랑스 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('모구리야영장', '서귀포모구리야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('독립기념관제3전시관', '라마다스위츠거제호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('풀빌라소랑', '아르떼스파앤풀빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하귀농협하나로마트', '더스테이센추리호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('만장굴', '서머셋 제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이호테우해수욕장', '동양콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프로스힐 제주', '프로스힐제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('안녕김녕SEA펜션', '안녕김녕씨')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('달콤한소금만들기', '달콤한소금만들기펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주서부신협한림지점', '스타즈호텔 제주로베로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('저지문화예술인마을', '스타즈호텔 제주로베로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울대학교연건캠퍼스/대학원기숙사', '벼리게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼례문화예술촌종합세미나실', '한나스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('옹암해수욕장', '장봉도 옹암파라다이스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('명가춘천막국수', '골든파크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스제주 오렌지동', '휘닉스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리솜스파캐슬', '스플라스 리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베드스테이션 선릉점', '베드스테이션호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더베스트성산', '더베스트제주성산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더그랜드섬오름 본관', '더그랜드섬오름호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수앤수호텔', '수앤수리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서귀포시모구리야영장', '서귀포모구리야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데제주리조트아트빌라스', '롯데리조트제주 아트빌라스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신화관 제주신화월드 스카이풀', '신화관 제주신화월드 호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('낙천주의자들', '호텔낙천주의자들')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포리별 독채', '서담미가')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청량산호불사', '빈티지1950')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연오정', '호텔노블리아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GOnae1101', '고내1101')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('곽지해변', '빅썸호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('육영수여사생가', '어썸스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SKYPARK호텔제주', '호텔스카이파크 제주1호점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오기목할망네', '오기목할망집')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금능해수욕장 아영장', '')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('일과1어촌종합센터', '제주공감게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대영빌딩', '농땡이 연구소')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼일공업고테니', '호텔골든데이지 서귀포오션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블루하와이호텔', '블루하와이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주월드컵경기장관광안내소', '또랑게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립한경대학교/제1공학관', '제주놀멍펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('중앙시장', '아루미호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오산역', '원월드 조이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나전칠기체험관', '탐라스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여미지식물원', '제주신화월드호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주 릴렌하우스', '릴렌하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('벨라유나', '벨라유나 인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무창포종합관리사무소무창포여름파출소농협365자동코너', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('왕대사', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('최참판댁', '서귀맨션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('히든클리프호텔&네이쳐', '히든클리프호텔앤네이쳐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SCARBORD호텔', '스카브로 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자제주호텔', '라마다프라자 제주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파크선샤인', '파크선샤인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('선샤인호텔', '파크선샤인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주항연안여객터미널', '토스카나 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('목포연안여객선터민러', '토스카나 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블루하와이호텔', '블루하와이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('안면암대연당7층대탑', '블루하와이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('꽃지해수욕장', '블루하와이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유탑유블레스호텔', '유탑유블레스호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해비치호텔앤리조트', '해비치호텔앤리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션스퀘어바', '오션스퀘어')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('밸류호텔', '서귀포JS호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수선사', '서귀포JS호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데캐슬프레지던트', '스타크호텔 제주로베로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('명월리백난아기념관', '브라운스위트제주 호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해수스파호텔COZA', '코자호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나비호텔제주', '제주해군호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연동한일시티파크', '하귤당')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼강빌딩', '베니키아호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄피하우스', '협재 하울 스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('디아일랜드블루호텔', '제주디아일랜드블루호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('점보빌리지', '글로스터호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('문화공원', '그린게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('케니스토리호텔', '호텔 휴식 서귀포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스제주섭지코지 수영장', '휘닉스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다누리도서관단양관광관리공단', '휘닉스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('월정비취', '월정스태이 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('협재 마당게스트하우스', '마당게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주세계문화엑스포공원첨성대영상관', '하루앤하루')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('회기역1번출입구', '하루앤하루')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더본호텔', '호텔더본')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('강릉시외버스터미널', '동양콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대박빌', '동천호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알루하우스 성산점', '알루하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더그랜드섬오름 본관', '더그랜드섬오름호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고향의맛손칼국수', '초롱민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데시티호텔 제주 수영장', '롯데시티호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신화호텔 랜딩광', '랜딩관 제주신화월드 호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트제주', '한화리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블랑드오조', '오조별장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수렌트카', '탱자싸롱 게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('강릉아이스아레나', '탱자싸롱 게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연포리조트연수원', '라마다프라자제주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('난지한강공원잔디마당', '이디게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웰호텔', '썬랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해운대해수욕장', '코너스톤호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔코너스톤', '코너스톤호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('흑사돈 표선점', '조은민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('사직공원', '히든클리프호텔&네이쳐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전주역', '호텔화인 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('화인관광호텔', '호텔화인 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그레이캐슬펜션', '완도네시아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨타워빌딩', '캠퍼트리 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('초롱민박조식', '초롱민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주황토리조트', '노을게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('풀벗', '풀벗아그리투리스모')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('두채하우스', '두채스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄 제주', '소노캄제주소노호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메리어트호텔', '제주메리어트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주 신화 메리어트호텔', '제주메리어트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스쿠버스토리 &게스트하우스', '스쿠버스토리게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고불락게스트하우스', '솔트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주난타파크', '호텔난타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원추추파크네이쳐빌', '하이원추추파크 오토캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('산방댁', '게스트하우스에코')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔케니모슬포', '호텔케니')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경희하우스', '제이르블랑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신성리조', '씨오르리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('최다선리조트', '취다선리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('취다선리조트 Tea&Meditation', '취다선리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주중문특급호텔리노베이션및증축사업', '파르나스호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이제주호텔', '신라스테이 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('향', '온더스톤게스트하우스 성산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('온더스톤게스트하우스', '온더스톤게스트하우스 성산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('온더스톤스위트', '온더스톤게스트하우스 성산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에코티엘오피스텔', '에코티엘')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('끌림36.5', '끌림365')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이치와이초이호스텔', 'HY팰리스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오드리인제주호텔', '오드리인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('샹그릴라호텔', '제주샹그릴라호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해양경찰청제주수련원', '제주지방해양경찰청 수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('The barn', '제주섬마리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봄그리고가을리조트B동', '봄그리고가을리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨유봄그리고가을리조트점', '봄그리고가을리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('최다선리조트', '취다선리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해비치호텔앤리조트 실외수영장', '해비치호텔앤리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)피엠씨프러덕션', '호텔난타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('난타호텔', '호텔난타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하쿠다 in Jeju (달곰)', '하쿠다 인 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하쿠다 in Jeju', '하쿠다 인 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다바이윈덤 제주더함덕호텔', '라마다호텔함덕')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CU제주메르블루점', '메르블루 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔RAOM', '라움호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('EKONOMYHOTEL', '코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스테이공간', '담양리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('VTOPHOTEL', '유탑부티크호텔앤레지던스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리솜포레스트', '포레스트리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('순천시청소년수련원본관', '순천시유스호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캬슬더아트', '어반스테이 여수웅천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('월드파크', '저스트스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한려파크', '호텔JCS여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파머스빌리지수영장', '파머스빌리지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프랑스모텔', '엠오르트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어은돌힐링캠핑장(청소년야영장)', '어은돌힐링캠피장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('돌섬참붕어찜', '라포레 메종')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노라무인호텔', '노라무인텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELYAM', '호텔암')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('송호관광지', '송호국민관광지 캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('화인프라자', '그랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GS칼텍스대한통운군산주유소', '베스트웨스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코자 스테이', '가온리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELAURA', '아우라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('단양대명콘도', '소노문 단양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대한불교조계종제5교구법주사템플스테이', '법주사템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하얀파크텔', '스테이 사계')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비지니스모텔', '베스트인시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더포인트호텔', '티포인트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한국투자상호저축은행', '호텔케니 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리솜스파캐슬', '스플라스 리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캬슬더아트', '어반스테이 여수웅천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔헤이본', '헤이븐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨이게스트하우스', '전부한옥마을웨이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔러브홀릭', '호텔야자 광주역점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립변산자연휴양림A동', '국립변산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('바다가되고싶어요', '스테이 무요일')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔로로시', '도로시모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GS칼텍스대한통운군산주유소', '베스트웨스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('RECENT모텔', '호텔노크온')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('뉴그랜드모텔사우나', '리본호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마곡사성보박물관', '마곡사템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELTHESOL', '더솔호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동뒤버스정류장', '비체펠리스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동춘 게스트하우스', '동춘게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SONOCALM', '소노캄 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('윈드빌가족호스텔', '라마다프라자 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('모텔씨네마', '씨네마모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전주한옥빌리지', '어라이브전주호텔시화연풍')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BOOHOTEL', '부호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('담양메타포토리아호텔신축공사', '호텔드몽드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('clemare', '디마레풀스파')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('태평양스파찜질방24시', '태평양스파찜질방')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제천 제이앤비 무인텔', '제천제이앤비')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('모텔러빙유', '라베호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대각산', '세봄빌')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그레이캐슬펜션', '유탑마리나호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('guesthouse모텔', '')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GS칼텍스대한통운군산주유소', '베스트웨스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덕유산국립공원남덕유분소', '호텔머드린')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('죽림동성당', '유탑마리나호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데부여리조트', '롯데리조트 부여')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여남초유송분교장폐교', '시공간펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('천왕이펜하우스4단지아파트', '소노문 단양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨사이드아덴', '단양관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('민둥산역', '힐링캠프펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('구름다리', '하늘에')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('농공읍행정복지센터', '모텔하늘하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('원일교회', '모텔하늘하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('단양군산림조합톱밥공장', '포레스트힐링캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('환상의바다호텔', '환상의바다리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('완주힐조타운바이오허브플러스', '원조힐조타운')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('당항포자연사전시관', '이끌림호텔 아비숑별관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('두륜산약수터', '월드펜션모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라온수상레져타운', '라온캠피장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금강레져타운24시간사우나찜질방', '금강레저타운')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('보현경로당', '와이누리봄게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('늘송파크텔', '첼로네')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라한호텔', '라한호텔 전주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('좌구산캠핑공원', '좌구산오토캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('YAM모텔', '스테이큐 용문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('보성군천연염색공예관', '한옥숨')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주센트럴파크미니미니랜드', '전주호텔꽃심')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리솜오션캐슬', '아일랜드 리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청주역', '전주호텔꽃심')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베이원파크빌딩', '여수베이원파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고흥썬밸리콘도미디엄', '고흥썬밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주흘수퍼', '메타펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('평택라마다앙코르호텔', '더포힐스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마검포태안캠핑장 입구,안내소', '마검포태안캠피장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대붕빌딩', '고추잠짜리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('괴산극장', '괴산자연드림파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수스테이비지니스호텔', '여수스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청송수퍼', '산마루펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('TOPCLOUD호텔', '탑클라우드 군산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('구리농수산물도매시장', '정우네펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무주향로산자연휴양림은하빛휴양관', '무주향로산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('가학산자연휴양림가학산동물농장', '흑석산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('허브아일랜드', '초락나루펜션앤글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('후즈넥스트 (게스트하우스)', '후즈넥스트게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('억수파크', '정글게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('조은한의원', '낭만게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)여수드론교육원', '그린글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전주덕진공원관리사무소', '갤럭시모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('군산고속버스터미널', '해오름민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('내장산문화광장', '정읍시 국민여가캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KW컨벤션센터', '토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('드라이브인한옥호텔달빛소리', '달빛소리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('OCEAN스테이폴리오호텔', '오션스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전등사', '호텔메세지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('콩나물박물관', '몽산포하얀집')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한국도로공사수목원', '몽산포하얀집')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에버랜드 홈브리지캐빈호스텔', '루파니애견캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('힐링호텔', '레지던스호텔라인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유탑유블레스', '여수유탑유블레스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('중앙출장소', '오늘이펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('목포근대역사관1관', '포레스트리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고시텔게스트하우스', '엘파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전쟁기념관', '오로라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수엑스포역버스정류장', '소노벨 변산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('의림지뉴인벤트홀', '리순덕호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더프라우드', '스테리움 제천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('풍수원성당', '라마다프라자 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('홍지문터널관리사무소', '소노캄 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('정달잠원GUESTHOUSE', '달잠게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울특별시서천연수원창의관', '서울시서천연구원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봄날', '풀빌라봄날')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전주해장국', '꾸레게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('익산빌딩', '블루보트게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('로니관광 호텔', '로니관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한국투자상호저축은행', '호텔케니 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('상구빌딩', '장고펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CURRAN비지니스모텔', '어반호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하남 Hotel stay35', '호텔스테이35')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('학동숙박시설', '벨라지오호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('게스트하우스짝', '더짝게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('VIGOCIELO', '비고리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('희리산자연휴양림관리사무소', '희리산자연휴양림자동차야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('부산역--->라마다앙코르 부산역호텔', '라마다앙코르 부산역호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웰파크시티 그린스토리', '힐링카운티')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고창웰파크시티주택홍보관', '석정온천휴스파')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('문헌서원', '문헌전통호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('가인당', '한옥마을숙박가인당')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('증심사', '증심사템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비지니스관광호텔', '탑클라우드호텔 익산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 U5', '호텔유파이브')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄거제주차타워', '소노캄 거제')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나주빛가람호텔', '빛가람호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('내장산국립공원', '내장산국립공원 내장아영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('ACC디자인호텔', 'ACC호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유토피아가족관광호텔', '유토피아가족호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('WHOTEL', '호텔W')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수예술랜드 수항도', '여수예술랜드리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쉼 한옥스파', '쉼한옥스파')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('태조궁', '태조궁호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('TOMI&SPA', '토미앤스파펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메리머드카페테리아', '메리머드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('김명옥숙박문화관', '경기전별당채')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연포리조트연수원', '센티마르펜션 글램핑캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토종가든', '나무와새')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 제주시 조천읍 신북로 577':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('소노벨', '소노벨 제주')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 제주시 서사로 129':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('소노벨', '아스타호텔')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '전북 무주군 설천면 무설로 1482':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('소노벨', '섬머그리스게스트하우스')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 제주시 조천읍 신북로 577':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('한강시민공원여의도지구', '라마다전주호텔')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '전북 전주시 완산구 팔달로 227':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('한강시민공원여의도지구', '오로라호텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '대전 유성구 엑스포로123번길 33':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 대전')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '울산 남구 삼산로 204':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 울산')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '서울 중구 삼일대로 362':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 명동')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '서울 마포구 마포대로 109':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 마포')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경북 경주시 보문로 422':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('코모도호텔', '코모도호텔 경주')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '부산 중구 중구로 151':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('코모도호텔', '코모도호텔 부산')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 서귀포시 성산읍 성산등용로17번길 35':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('썬라이즈호텔', '썬라이즈호텔 성산')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '강원 속초시 청초호반로 291':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('썬라이즈호텔', '썬라이즈호텔 속초')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경남 하동군 화개면 쌍계로 532-6':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('켄싱턴리조트', '켄싱턴리조트 하동')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '전북 남원시 소리길 66':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('켄싱턴리조트', '켄싱턴리조트 남원')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '충북 충주시 앙성면 산전장수1길 103':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('켄싱턴리조트', '켄싱턴리조트 충주')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '제주도 제주시 한림읍 한림해안로 530':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('켄싱턴리조트', '켄싱턴리조트 제주한림')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '제주도 제주시 구좌읍 월정1길 79-13':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('켄싱턴리조트', '달물인연')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경기 파주시 탄현면 필승로 448':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('G7호텔', '호텔G7')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '부산 부산진구 황령대로17번길 21':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('G마루호텔', '마루호텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '서울 종로구 수표로18가길 8':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('IMT모텔', '종로아이엠티호텔')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '인천 중구 영종해안남로321번길 208':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('그랜드하얏트호텔', '그랜드하얏트인천')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '서울 용산구 소월로 322':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('그랜드하얏트호텔', '그랜드하얏트서울')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경기 가평군 가평읍 잎너비길 17-66':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라온빌', '가평라온빌스파앤독채펜션')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '서울 용산구 소월로 322':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라온빌', '호텔더그랑 중문')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '서울 중구 삼일대로 362':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 명동')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '서울 마포구 마포대로 109':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 마포')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '울산 남구 삼산로 204':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('롯데시티호텔', '롯데시티호텔 울산')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('2006증축신고40호', '해오름하우스펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('3917마중', '오션스위츠 제주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('9BRICKHOTEL', '나인브릭호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('AVENUE186호텔', '호텔에비뉴186')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Bellino호텔', '벨리노S호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BK호텔', 'BK호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BLUESPRING호텔', '블루스프링부띠끄호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BROWNDOTHOTEL', '브라운도트호텔 송도해수욕장점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운돈트호텔', '브라운도트호텔 경주불국사점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 울산 정자해수욕장점', '브라운도트 정자해수욕장점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CAPPUCCINO호텔', '호텔카푸치노')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CS호텔', 'CS모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('CU', '여수비치펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('종화동4631제2종근린생활시설', '여수비치펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Dukstay', '덕스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('EFK알레그리아호텔', 'FK알레그리아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알레그리아호텔서귀포', 'FK알레그리아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('E모텔', '이븐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H.AVENUE호텔', 'H에비뉴 성신여대점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이치에비뉴 강릉경포', 'H에비뉴 강릉경포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HAVENUE관광호텔', '에이치에비뉴호텔 건대점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELHAY', '호텔헤이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTELLAYERS', '레이어스호텔 하단점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTEL화엄267', '호텔화엄267')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HY팰리스호텔', 'HY초이호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HY펠리스호텔', 'HY초이호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('MH호텔 별관', 'MH호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('IBIS앰배서더호텔', '이비스앰배서더 수원점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JHOUSE', '펜션J하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JS프라이빗타운', '제주 까르마')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메리어트모텔', '녹스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JW 메리어트 호텔 서울', 'JW메리어트호텔 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JW메리어트호텔', 'JW메리어트동대문스퀘어서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KT경주수련관 507호', 'KT경주수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KT경주수련관', 'KT경주수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Kt대관령수련관', 'KT대관령수련관')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('BK호텔', 'BK호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SL호텔강릉', 'SL호텔 강릉')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('sl호텔강릉', 'SL호텔 강릉')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('SOL호텔', '솔게스트하우스 양양서핑점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('STX리조트', 'STX문경리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('WE호텔', 'WE호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('yaja 포항여객터미널', '호텔yaja 포항여객터미널점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('강릉게스트하우스', '강릉게스트하우스 커피거리점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('게으른소나기', '게으른소나기게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경포비치관광호텔', '경포비치호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인별', '곁겹')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고고애월', '고고애월게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('곤지암리조트EW빌리지', '곤지암리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('골든튤립에버 용인호텔', '골든튤립에버용인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('공주한옥마을공주관풍정', '공주한옥마을')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립아세안자연휴양림', '국립아세안자연휴양림 브루나이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드조선제주호텔', '그랜드조선 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드하얏트인천 레스토랑8', '그랜드하얏트인천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드하얏트인천 웨스트타워', '그랜드하얏트인천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드하얏트 인천 웨스트타워', '그랜드하얏트인천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드팰리스 호텔 인천', '그랜드팰리스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금등리', '금등이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금호제주리조트', '제주금호리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나인스파빌', '나인스파빌리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나인트리프리미어호텔', '나인트리프리미어호텔 명동2')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노떼마리아호텔', '노떼라미아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노마드인제주', '노마드인 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노보텔앰배서더서울동대문호텔&레지던스', '노보텔앰배서더 서울동대문호텔&레지던스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노보텔엠베서더부산', '노보텔엠베서더 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다인오세아노 호텔', '다인오세아노호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('달무리 스테이', '달무리스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('달이차오른다가자 게스트하우스', '달이차오른다가자게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('담모라', '담모라리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대명리조트오션파크', '대명리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대천경찰수련원', '대천경찰연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대호팬션타운A동', '대호팬션타운')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더그랜드 섬오름 호텔', '더그랜드섬오름호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더그랜드섬오름 신관', '더그랜드섬오름호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더베스트성산제주호텔', '더베스트제주성산호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더베스트제주성산', '더베스트제주성산호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더스위트호텔', '더스위트호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더스테이송도', '더스테이 송도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더케이지리산가족호텔그랜드볼룸', '더케이지리산가족호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덕구온천관광호텔', '덕구온천리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('데미샘자연휴양림운향채', '데미샘자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덕유산리조트햇님동', '덕유산리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('데이나이스호텔', '데이나이스호텔 대천지점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타호텔 헤리티지 경주점', '덴바스타호텔헤리티지 경주점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동강시스타', '동강시스타리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동강시스타스파시설', '동강시스타리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('디아크리조트히든호텔동', '디아크리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라LakaisSANDPINE', '라카이샌드파인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다', '라마다용인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다 태백호텔', '라마다태백호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라메르', '라메르호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라온호텔', '라온호텔&리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라테라스리조트앤스파 코브스위트', '라테라스리조트앤스파')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레스트리 리솜', '레스트리리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레스트리리솜 마묵라운지', '레스트리리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데시티호텔 대전 대전', '롯데시티호텔 대전')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데호텔부산', '롯데호텔 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('루프탑정원', '루프탑정원 게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리젠시', '리젠시모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마리나베이속초', '마리나베이 속초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마티에오시리아호텔 수영장', '마티에오시리아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메르블루호텔', '메르블루 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('메이모텔A동', '메이모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('명성아카데미하우스샬롬관', '명성아카데미하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('몬드리안 서울 이태원', '몬드리안 서울이태원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('문라이트', '문라이트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('문성자연휴양림민들래', '문성자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('문성자연휴양림원추리', '문성자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('물뜰에쉼팡리조트A동', '물뜰에쉼팡리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('물뜰에쉼팡리조트관리동', '물뜰에쉼팡리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('미드나잇 인 경주', '미드나잇인경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('민수민수 게스트하우스', '민수민수게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('밀레니엄 서울힐튼', '밀레니엄서울힐튼호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('바라는바다', '바라던바다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('바람의언덕', '바람의언덕리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('배다리생태공원', '힐튼가든인 서울강남점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('백년한옥', '백년한옥 게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('백운산휴양타운', '백운산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베니키아호텔', '베니키아호텔 산과바다대포항')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베어스타운', '베어스타운리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베이힐풀앤빌라A동', '베이힐풀앤빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('벤티모호텔앤레지던스', '벤티모호텔앤레지던스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('부산세관 수련원', '부산세관수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운 스위트 제주 호텔', '브라운스위트제주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블랙스톤골프&리조트', '블랙스톤리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블랙스톤골프&리조트클럽하우스', '블랙스톤리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블루힐', '블루힐펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블룸호텔', '블룸호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비너스텔', '브라운도트호텔 포항여객터미널점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비스타워커힐서울', '비스타워커힐 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비체팰리스', '비체펠리스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼성하이빌', '홍대딸기핑크게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼화사', '삼화사템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('상관리조트&스파', '상관리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새왓댁', '새왓댁 1호점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('샐리스 제주', '샐리스제주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서머셋 제주신화월드', '서머셋제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울숲공원', '라마다제주시티홀')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서웅', '노블레스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서천유스호스텔힐링B동', '서천유스호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('석모도자연휴양림 산림문화휴양관', '석모도자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('섬게스트하우스&카페', '섬게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('성베네딕도회왜관수도원마오로플라치도관', '성베네딕도회왜관수도원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('성산스테이션', '성산스테이션게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('센터마크호텔', '아미드호텔 서울')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소금막호스텔', '소금막리조트')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '충남 보령시 오천면 오천해안로 709-3':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('힐하우스', '힐하우스 보령')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '대전 서구 장안로 588':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('힐하우스', '힐하우스 대전')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '경기 양평군 강하면 강남로 489':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('힐하우스', '힐하우스 양평')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '대구 중구 서성로 102-6':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('호텔여기어때', '호텔여기어때 대구점')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '강원 강릉시 해안로406번길 13-9':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('호텔여기어때', '호텔여기어때 강릉점')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '서울 영등포구 국회대로68길 24':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('호텔더디자이너스', '호텔더디자이너스 영등포')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '인천 남동구 남동대로765번길 8':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('호텔더디자이너스', '호텔더디자이너스 인천')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '인천 부평구 경원대로1417번길 18-8':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('피아노호텔', '피아노호텔 인천')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '대전 동구 동서대로1683번길 94-4':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('피아노호텔', '피아노호텔 대전')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경북 경주시 태종로727번길 31':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('청춘게스트하우스', '청춘게스트하우스 경주')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '제주도 서귀포시 이중섭로 31':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('청춘게스트하우스', '청춘게스트하우스 제주')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '경기 남양주시 다산지금로36번길 21-6':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('에이원호텔', '에이원호텔 남양주')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '제주도 제주시 사장1길 10':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('에이원호텔', '에이원호텔 제주')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '전남 구례군 산동면 하관길 49-15':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('쉴모텔', '마리호텔')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('히든클리프호텔', '히든클리프호텔앤네이쳐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('히든클리프 호텔', '히든클리프호텔앤네이쳐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('히든클리프호텔&네이쳐', '히든클리프호텔앤네이쳐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스제주섭지코지 힐리우스', '휘닉스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('후즈넥스트게스트하우스', '후즈넥스트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('황룡 stay', '황룡원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('홍류동탐방지원센터', '무주덕유산리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('화왕산자연휴양림관리사무소', '화왕산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('홀리데이 인 광주호텔', '홀리데이인광주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔크라페', '호텔크라페 경산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔케이월드', '호텔케이월드 광진')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔컬리넌', '호텔컬리넌 광진')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔춘향가', '호텔춘향')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔위드 제주', '호텔위드제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔스카이파크3', '호텔스카이파크 명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔스카이파크 명동3호점', '호텔스카이파크 명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔샬롬카페', '호텔샬롬제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔마띠유', '호텔마띠유 여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔더그랑중문', '호텔더그랑 중문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔YAM 영천터미널점', '호텔얌 영천터미널점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔YAJA', '호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('현대해상연수원', '현대해상 변산연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('현대자동차블루핸즈', '브릿지레지던스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('현대수콘도미니엄B동', '현대수콘도미니엄')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해비치호텔앤드리조트 제주', '해비치호텔앤리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('합천HC호텔', '합천 HC모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('함양남계서원(사적제499호)', '자인빌리지펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('함박골수박한마음센터', '함박골마을영농조합법인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화호텔앤드리조트 여수', '한화호텔앤드리조트 여수벨메르')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트설악설악', '한화리조트설악')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 해운대 해운대', '한화리조트 해운대')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한옥호텔영산재', '한옥호텔궁')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한림속하루', '한림속 하루')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원 추추파크', '하이원리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하오호텔', '하오모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하버호텔', '하버하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플레이스캠프제주', '플레이스캠프 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플레이스 캠프', '플레이스캠프 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플래밀리풀빌라호텔A동', '플래밀리풀빌라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플래밀리풀빌라호텔C동', '플래밀리풀빌라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프라임호텔', '프라임모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('평대리 943 Tu Casa', '평대리943 Tu Casa')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('평대리943', '평대리943 Tu Casa')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파인아트라벨', '파인아트라벨호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('털보네 게스트 하우스', '털보네게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탐무인모텔', '탑모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탐라스테이호텔제주', '람타스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('타시탈레게스트하우스', '타시탈레 게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽멤피스글램핑앤리조트', '클럽멤피스 글램핑앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('퀸즈 드라이브인 (퀸즈 무인텔)', '퀸즈모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코모도호텔 부산 부산', '코모도호텔 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코모도호텔 부산 부산 부산', '코모도호텔 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코모도호텔 경주 경주 경주 경주', '코모도호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코모도호텔 경주 경주 경주 경주 경주', '코모도호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코레일낙산연수원', '코레일 낙산연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코너스톤주식회사', '코너스톤호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코델리아리조트', '코델리아S호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트설악밸리', '켄싱턴리조트 설악밸리')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴호텔여의도', '켄싱턴호텔 여의도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트 하동 지리산하동', '켄싱턴리조트 하동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캠퍼트리 호텔', '캠퍼트리 호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캠퍼트리호텔&리조트', '캠퍼트리 호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청송슈퍼', '산마루펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청계산역입구', '오라카이청계산호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('천은사(전라남도문화재자료제35호)', '천은사템플스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('천지연크리스탈호텔', '천지연호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('책속에풍덩', '책속에 풍덩')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주식회사인터투어', '단양관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주식회사빌리스코리아중문', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주문진해수욕장', '전주역모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('주문진호텔', '주문진리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('종화동4631제2종근린생활시설', '여수비치펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주펄관광호텔', '제주펄호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주참??가마', '제주참숯가마')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주올레여행자센터', '올레스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주옥게스트하우스', '제주옥')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 5단지', '제주신화빌라스 5단지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주 신화빌라스 5단지', '제주신화빌라스 5단지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화빌라스 5단지', '제주신화빌라스콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신라호텔 풀사이드바', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신라호텔 수영장', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신라 피트니스센터', '제주신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주스테이인성산', '제주스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주샹그릴라호텔앤리조트 식당', '제주샹그릴라호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주라마다앙코르성산호텔', '라마다앙코르서귀포호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주공항게스트하우스 웨이브사운드', '제주공항게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주게토게스트하우스파티', '제주게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전주 A+호텔', '전주 A+ 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전통학습체험장', '정선 개미들마을펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('자유인의 카라반', '자유인의카라반')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('자몽', '자몽호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주 자몽호텔', '자몽호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인터컨티넨탈서울코엑스', '인터컨티넨탈 서울코엑스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이스턴호텔 제주', '이스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스스타일앰배서더 서울강남', '이비스앰배서더 서울강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스스타일 앰배서더 명동', '이비스앰배서더 서울명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이마트24산정호수점', '산정호수 가족글램핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('충남 아산시 시민로 377', '부산장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유탑부티크호텔&레지던스', '유탑부티크호텔앤레지던스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유탑마리나호텔&리조트', '유탑마리나호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('유니호텔 앤 풀빌라', '유나호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스틴조선호텔 부산점 수영장', '웨스틴조선호텔 부산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스트힐스 프라이빗 풀빌라', '웨스트힐스호스텔여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스던그레이스호텔', '웨스턴그레이스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('월정리썬씨티 게스트하우스', '월정리썬씨티게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('울릉벚꽃 게스트하우스', '울릉벚꽃')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용평리조트그린피아콘도', '용평리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용평리조트 그린피아콘도', '용평리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용평리조트 드래곤밸리호텔', '용평리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용평리조트 빌라콘도', '용평리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('용문산자연휴양림', '용문산야영장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('온더스톤 게스트하우스', '온더스톤게스트하우스 성산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트클럽하우스', '오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트콘도B동', '오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트 골프콘도B동', '오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션투유리조트 속초설악비치호텔앤콘도', '오션투유콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션스타레저콘도', '오션스타콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션뷰제주 게스트하우스', '오션뷰제주게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주오리엔탈호텔', '오리엔탈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('영인산자연휴양림관리사무소휴양관', '영인산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수베네치아호텔&리조트', '여수베네치아호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠포토사진학원제일가정의원', '토마스앤레지나오피스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠에이치모텔신관', '엠에이치모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠에이치모텔별관', '엠에이치모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠스테이호텔제주', '엠스테이호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠버서더호텔', '엠배서더호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에코랜드호텔', '에코랜드 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이플러스호텔', '에이플러스모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('안동 그랜드호텔', '안동그랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('안녕함덕게스트하우스', '안녕함덕')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아일랜드 리솜(회사복지)', '아일랜드 리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스플라스리솜 스테이타워', '스플라스 리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스플라스리솜 플렉스타워', '스플라스 리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아인스호스텔', '아인스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아스티호텔 부산역', '아스티호텔 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아난티힐튼 부산', '아난티 힐튼 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨에스호텔&리조트제주', '씨에스호텔앤리조트제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬크루즈호텔&리조트', '썬크루즈호텔리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬쿠르즈리조트', '썬크루즈호텔리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬라이즈호텔 성산 성산 성산', '썬라이즈호텔 성산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬라이즈호텔 성산 성산', '썬라이즈호텔 성산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신화관 제주신화월드호텔앤리조트', '서머셋 제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신화관 제주신화월드 호텔앤리조트', '서머셋 제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신신호텔제주에어포트', '신신호텔 제주공항')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신신호텔 제주공항점', '신신호텔 제주공항')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신북온천호텔', '신북온천')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이 울산 울산', '신라스테이 울산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이 구로 구로', '신라스테이 구로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('시네마호텔', '시네마모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스타크호텔 제주로베로', '롯데시티호텔 마포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('슈어스테이플러스호텔식당', '슈어스테이플러스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('숨게스트하우스 제주공항점', '숨게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주도 서귀포시 성산읍 해맞이해안로 2688', '취다선 리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수원종합운동장축구장', '달에물들다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('수원월드컵경기장', '다인오세아노 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소링왓', '소랑왓')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노펠리체 빌리지델피노', '소노펠리체 델피노')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄거제', '소노캄 거제')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨 제주 제주 제주', '소노벨 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨 제주 제주', '소노벨 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨 변산(회사 휴양시설)', '소노벨 변산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노문단양 WEST TOWER', '소노문 단양')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '대구 동구 이노밸리로 7':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('', '앙코르호텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '부산 해운대구 구남로 9':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('', '라마다앙코르해운대호텔')
    elif fdf['ROAD_NM_ADDR'].iloc[i] == '전남 순천시 역전광장3길 44':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('', '순천24게스트하우스')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('테디밸리골프앤리조트자재관리실', '머큐어앰배서더 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)페이퍼플라자', '위코스테이 명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포포인츠 바이쉐라톤 서울구로', '포포인츠바이쉐라톤 서울구로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데시티호텔 명동 명동', '롯데시티호텔 명동')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨경주 오션플레이', '소노벨 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('THESUITESHOTEL', '스위트호텔경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('부산 아르반호텔', '아르반호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라발스스카이카페&바', '라발스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('LOULS호텔', '루이스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코모도호텔 경주 경주 경주', '코모도호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Yaja명서컨벤션', '호텔야자 창원명서컨벤션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스타즈호텔 울산점 울산점', '스타즈호텔 울산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('TAYHOTEL', '스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('MERCURE앱배서더호텔', 'AC호텔 바이 메리어트 강남')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('천안상록리조트그랜드홀', '천안상록호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('(주)신세계조선호텔부산', '웨스턴조선호텔 부산점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬라이즈호텔', '썬라이즈호텔 속초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('선재낚시공원', '노랑행궁')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('춘천시춘천시그랜드모텔', '춘천시그랜드모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('IFC부산오슬로애비뉴', '아바니센트럴부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스텐포드인부산호텔', '스탠포드인 부산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('패밀리게스트하우스', '패밀리하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KING모텔', '브라운도트호텔 마산양덕점')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프래밀리풀빌라호텔A동', '프래밀리풀빌라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프래밀리풀빌라호텔C동', '프래밀리풀빌라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('금란모텔', '라라비안코호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청초비치', '호텔청초')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('윌모텔', '호텔나무')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코모도호텔 경주 경주 경주', '코모도호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장유장유넘버25호텔', '장유넘버25호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('테스노래클럽', '베르사체모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨-라운지', '롯데시티호텔 대전')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장령산산림문화휴양관', '장령산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('데일리호텔', '모텔식스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베네치아관광호텔', '베네치아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청풍리조트 레이크호텔', '레이크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이에스리조트클럽', '클럽이에스 제천리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('여수하이오션', '여수유탑유블레스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('USUNGHOTEL', '유성관광호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('HOTEL535TAY', '호텔스테이53')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장군도빌딩', '장군도 펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔만리', '말리호텔앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청풍리조트 레이크호텔', '청풍레이크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('장수도깨비전시관', '장수도깨비한옥앤글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스마트하우스오피스텔', '베스트웨스턴플러스전주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울시서천연구원', '산하에이치엠')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스테이틈', '스테이 틈')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔노블리아', '노블피아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔R&T', '호텔조이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 휴식 서귀포', '호텔휴식 서귀포')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스타즈호텔제주로베로구내', '호텔로베로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('산토리니게스트하우스', '히든스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔빠레브', '빠레브호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('별리게스트하우스&카페', '벼리게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('점보네게스트하우스', '애월 오션스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔대동', '대동호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('윈스카이호텔', '호텔윈스카이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔휘슬락', '휘슬락호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레프트핸더 게스트하우스', '레프트핸더게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스위트호텔 제주점', '더스위트호텔 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스위트호텔남원', '더스위트호텔 남원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코업시티호텔 하버뷰', '코업시티호텔하버뷰')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('까사그란데&덕평본가', '까사그란데')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서머셋 제주신화월드', '서머셋제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신화월드 서머셋', '서머셋제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('빌라드애월뷰티크호텔', '탑스텐 빌라드애월')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑스텐빌라드애월 제주', '탑스텐 빌라드애월')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('일레인호텔', '일레인제주리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데호텔 제주', '롯데호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코업시티호텔 성산', '코업시티호텔성산')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인덱스 제이드림호텔', '플로라제이드림호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라온빌', '호텔더그랑 중문')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아지레스턴스호텔', '르메인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('르메인호텔 서귀포', '르메인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리메인호텔', '르메인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다인오세아노 호텔', '다인오세아노호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라온호텔&리조트', '라온프라이빗타운')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주야자원', '야자원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주 야자원', '야자원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더웰테라스', '스타즈호텔 제주로베로')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('진주논개시장', '타마라 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('도체비낭게스트하우스', '너랑나랑게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노형수퍼마?R', '노형수퍼마켙')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔xym', 'XYM모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스너글', '힐링스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('긋모닝하우스팬선', '굿모닝하우스')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '강원 삼척시 도계읍 심포남길 99':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('하이원리조트', '하이원리조트 삼척')
    elif fdf['VISIT_AREA_NM'].iloc[i] == '강원 정선군 고한읍 하이원길 265-1':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('하이원리조트', '하이원리조트 힐콘도')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 서귀포시 성산읍 성산중앙로 56-1':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('더포레스트호텔', '바라봄')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 서귀포시 1100로 453-95':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('스타즈호텔 제주로베로', 'WE호텔 제주')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 서귀포시 성산읍 성산중앙로 43':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('아따랑스 제주', '성산스테이션게스트하우스')

for i in range(len(fdf)):
    if fdf['ROAD_NM_ADDR'].iloc[i] == '제주도 제주시 구좌읍 구좌해안로 178':
        fdf['VISIT_AREA_NM'].iloc[i] = fdf['VISIT_AREA_NM'].iloc[i].replace('라마다호텔제주', '안녕김녕씨')

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다호텔&스위트 평창', '라마다호텔 평창')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔ORA', '호텔오라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스스타일앰버서더호텔', '이비스앰배서더호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파라다이스시티 원더박스', '파라다이스시티호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('송도 더센트럴파크호텔', '송도센트럴파크호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JUSTSLEEP호텔', '저스트슬립')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인부산서면', '토요코인호텔 부산서면')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('js호텔', '서귀포JS호텔')

df = fdf.copy()

# 겹치는 숙소 변수명 범주화
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25 수원화성행궁', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔그레이스리 서울', '호텔그레이스리')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 광화문', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('메이필드호텔 서울', '메이필드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('JW메리어트동대문스퀘어서울', 'JW메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('AC호텔 바이 메리어트 서울강남', 'AC호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('콘래드 서울', '콘래드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('노보텔앰배서더 서울용산', '노보텔앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 삼성', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그랜드워커힐서울', '그랜드워커힐')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야놀자천안성정점', '호텔야놀자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('글래드호텔 마포', '글래드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('이비스앰배서더 서울명동', '이비스앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('보코서울강남', '보코')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 서초', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 송도', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더캡슐 명동점', '더캡슐')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔스퀘어 안산', '호텔스퀘어')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('에이치에비뉴호텔이대점', '에이치에비뉴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴호텔 평창', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그랜드하얏트인천', '그랜드하얏트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('엠블호텔킨텍스', '엠블호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('글래드강남코엑스센터', '글래드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('노보텔앰배서더 서울동대문호텔&레지던스', '노보텔앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 인천주안역점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데호텔 월드점', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔스카이파크 인천송도', '호텔스카이파크')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어거스트 청평', '어거스트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('안테룸 서울', '안테룸')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더리프 서울사당', '더리프')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 역삼', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데시티호텔 마포 마포', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 동탄', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데시티호텔 명동', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('나인트리프리미어호텔 명동2', '나인트리프리미어호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('안다즈 서울강남', '안다즈')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노휴 양평', '소노휴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다서울신도림호텔', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('글래드 여의도', '글래드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('송도센트럴파크호텔', '센트럴파크호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔더디자이너스 DDP점', '호텔더디자이너스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('파주 그라체호텔 금촌점', '그라체호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('비스타워커힐 서울', '비스타워커힐')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소피텔 앰배서더 서울', '소피텔앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그래비티 서울 판교 오토그래프컬렉션', '그래비티')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('L7 홍대', 'L7')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('위코스테이 명동', '위코스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소테츠프레사인 서울명동', '소테츠프레사인')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베스트웨스턴프리미어 인천에어포트호텔', '베스트웨스턴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('조선팰리스 서울강남', '조선팰리스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('골든튤립에버용인호텔', '골든튤립호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔더디자이너스 서울역점', '호텔더디자이너스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아미드호텔 서울', '아미드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('머큐어 앰배서더 서울 홍대', '머큐어앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('오라카이송도파크호텔', '오라카이파크호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔 더 디자이너스 홍대', '호텔더디자이너스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('이비스 스타일 앰배서더 서울 용산', '이비스앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스탠포드호텔 서울', '스탠포드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('나인트리호텔 명동', '나인트리호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('L7호텔강남', 'L7호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코트야드 바이메리어트 서울타임스퀘어', '코트야드바이메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('도화동비지니스호텔신라스테이', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신천 A+ 무인호텔', '신천A')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 천안', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다용인호텔', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('노보텔 스위트 앰배서더 서울 용산', '노보텔앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('피아노호텔 인천', '피아노호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인호텔 인천부평점', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 종로점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더스테이트 선유', '더스테이트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔스카이파크 명동', '호텔스카이파크')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('서머셋팰리스 서울', '서머셋팰리스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 영등포점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔아트리움 종로', '호텔아트리움')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('필스테이 이화부띠끄점', '필스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다프라자 수원호텔', '라마다프라자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인호텔 영등포점', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('페어몬트 앰배서더 서울', '페어몬트앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('쉐라톤그랜드 인천호텔', '쉐라톤그랜드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('이비스앰배서더호텔 인사동', '이비스앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('로이넷호텔 서울마포', '로이넷호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더스테이 송도', '더스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그랜드하얏트서울', '그랜드하얏트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔 나루 서울 엠갤러리', '호텔나루')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔케이월드 광진', '호텔케이월드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트 평창', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('도미인서울강남호텔', '도미인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔컬리넌 종로', '호텔컬리넌')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('파크하야트서울호텔', '파크하야트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('홀리데이인익스프레스 서울홍대', '홀리데이인익스프레스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('야코리아호스텔 강남점', '야코리아호스텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('이벤트리종로호텔', '이벤트리호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마라앙코르김포한강호텔', '라마라앙코르')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스타즈호텔 명동2호점', '스타즈호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔8월 천호', '호텔8월')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('나인트리호텔 동대문', '나인트리호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더휴식 이로호텔 월곶지점', '더휴식이로호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25호텔 양주송추점', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('이비스앰배서더 수원점', '이비스앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('오크우드프리미어 인천', '오크우드프리미어')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노캄제주소노호텔앤리조트', '소노캄')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔더디자이너스 동대문', '호텔더디자이너스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 구로', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 서대문', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('여주마조렐글램핑리조트', '마조렐글램핑리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('머스트스테이호텔 명동점', '머스트스테이호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25 인천신포점', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔세븐스텝 시흥정왕점', '호텔세븐스텝')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('힐튼가든인 서울강남점', '힐튼가든인')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('포포인츠 바이 쉐라톤 조선 서울명동', '포포인츠바이쉐라톤')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('24게스트하우스 잠실점', '24게스트하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다프라자 제주호텔', '라마다프라자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데시티호텔 마포', '롯데시티호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('데이즈호텔앤스위트 인천에어포트', '데이즈호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 모란역점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('김포 아뮤즈 호텔', '아뮤즈호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25 영등포점', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더스테이트 선유 ', '더스테이트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('인터컨티넨탈 서울코엑스', '인터컨티넨탈')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더 디자이너스 리즈강남프리미어', '호텔더디자이너스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 포천송우점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('JW메리어트호텔 서울', 'JW메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔아벤트리 종로', '호텔아벤트리')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 동인천역점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 종로점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔더캐슬 2호점', '호텔더캐슬')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔 스카이파크 킹스타운 동대문점', '호텔스카이파크')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('포포인츠바이쉐라톤 서울구로', '포포인츠바이쉐라톤')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('영등포코모드모텔', '코모드모텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 인천송도점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('시그니엘 서울', '시그니엘')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인호텔 서울동대문2', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코트야드 바이 메리어트 수원', '코트야드바이메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25 광명사거리점', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이동탄', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('몬드리안 서울이태원', '몬드리안')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴호텔 여의도', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('H에비뉴 성신여대점', '에이치에비뉴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('노보텔 앰배서더 서울 강남', '노보텔앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('나인트리프리미어호텔 서울판교', '나인트리프리미어호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아호텔 제주', '베니키아호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다인천호텔', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25 방이점', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('나인트리프리미어호텔 인사동', '나인트리프리미어호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 가평', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('에이치에비뉴호텔 건대점', '에이치에비뉴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스타즈호텔 독산', '스타즈호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다호텔 서울동대문', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('마리나베이 속초', '마리나베이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('파라다이스호텔 부산', '파라다이스호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('쏠비치 양양', '쏠비치')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데호텔 부산', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데호텔 울산', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('체스터톤스속초', '체스터톤스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 해운대', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('파크하얏트 부산', '파크하얏트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스카이베이 경포', '스카이베이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('오게 부산역점', '오게')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스탠포드인 부산', '스탠포드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아난티 남해', '아난티')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('안나티남해컨트리클럽', '아난티')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('경주힐튼호텔', '힐튼호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트설악', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트 거제벨버디어', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('리버틴호텔 경주점', '리버틴호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('밸류호텔부산', '밸류호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('웨스턴조선호텔 부산점', '웨스턴조선호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('시그니엘 부산', '시그니엘')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인 동성로점', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아바니센트럴부산', '아바니센트럴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노벨 청송', '소노벨')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다앙코르 부산역호텔', '라마다앙코르')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('이비스앰배서더 부산해운대', '이비스앰배서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 송도해수욕장점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 성당못점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 수영점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인호텔 부산서면', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그랜드조선 부산', '그랜드조선')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('쏠비치 삼척', '쏠비치')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('페어필드 바이메리어트 부산', '페어필드바이메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 서면', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('KT&G상상마당 부산스테이', '부산스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아난티 힐튼 부산', '아난티')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('페어필드바이메리어트 부산송도비치', '페어필드바이메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('썬라이즈호텔 속초', '썬라이즈')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다앙코르해운대호텔', '라마다앙코르')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('하이오션 경포', '하이오션')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('대구메리어트호텔', '메리어트호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노캄 거제', '소노캄')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화호텔앤드리조트 여수벨메르', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('하이원리조트 삼척', '하이원리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 속초등대점', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트 해운대', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('해운대영무파라드호텔', '영무파라드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 마산양덕점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 하동', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인호텔 울산삼산점', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 울산', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('하운드시그니처호텔 해운대점', '하운드시그니처호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더케이호텔 경주', '더케이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다호텔 속초', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('휘닉스 평창', '휘닉스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('속초롯데호텔리조트', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄트호텔 광안리', '켄트호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('강릉게스트하우스 커피거리점', '강릉게스트하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 속초해변', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('웨스틴조선호텔 부산점', '웨스틴조선')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코모도호텔 부산', '코모도호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 포항여객터미널점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('레이어스호텔 하단점', '레이어스호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('힐튼호텔 경주', '힐튼호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브리드호텔 양양', '브리드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코모도호텔 경주', '코모도호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다앙코르정선호텔', '라마다앙코르')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 강릉경포점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라한호텔 포항', '라한호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔여기어때 대구점', '호텔여기어때')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데리조트 속초', '롯데리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 설악밸리', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베스트웨스턴플러스 경주', '베스트웨스턴플러스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔피코 통영지점', '호텔피코')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('리센오션파크속초', '리센오션파크')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('H에비뉴 강릉경포', '에이치에비뉴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('장유넘버25호텔', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소설재 첨성대점', '소설재')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 사상낙동대로점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 호텔 서면범천점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('경주게스트하우스 프렌드', '게스트하우스프렌드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('SL호텔 강릉', 'SL호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노벨 경주', '소노벨')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('노보텔엠베서더 부산', '노보텔엠베서더')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스위트호텔경주', '스위트호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토요코인호텔 부산해운대2호점', '토요코인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('넘버25 남포대청점', '넘버25')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 대구수성점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('시타딘커넥트호텔하리 부산', '커넥트부산호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('솔라리아니시테츠호텔 부산점', '솔라리아니시테츠호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔현대바이라한 울산', '호텔현대바이라한')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토모노야호텔앤료칸 경주점', '토모노야호텔앤료칸')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔더디자이너스 종로점', '호텔더디자이너스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 설악비치', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔리버사이드 울산', '리버사이드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 진주성점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스타즈호텔 울산점', '스타즈호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한산마리나리조트고성점', '한산마리나리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('덴바스타호텔헤리티지 경주점', '덴바스타호텔 헤리티지 경주점')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아스티호텔 부산', '아스티호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 장생포점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 원주기업도시점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('삼성호텔 거제', '삼성호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아호텔 산과바다대포항', '베니키아')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라한셀렉트 경주', '라한')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('청춘게스트하우스 경주', '청춘게스트하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아호텔해운대', '베니키아')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데시티호텔 울산', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔 케니 서귀포', '호텔케니')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 창원명서컨벤션', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔yaja 포항여객터미널점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('솔게스트하우스 양양서핑점', '솔게스트하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('청평포세이돈카라반', '포세이돈')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그랜드인터컨티넨탈서울파르나스', '파르나스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코드야드바이메리어트서울보타닉파크', '코트야드바이메리어트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('AC호텔 바이 메리어트 강남', 'AC호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노벨 변산', '소노벨')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('웨스트힐스호스텔여수', '웨스트힐스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('쏠비치 진도', '쏠비치')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노캄 여수', '소노캄')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다프라자 여수', '라마다프라자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노문 단양', '소노문')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('밀리오레호텔 명동', '밀리오레호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노벨 천안', '소노벨')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔마띠유 여수', '호텔마띠유')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 지리산남원', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔현대바이라한 목포', '호텔현대바이라한')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더스위트호텔 남원', '더스위트호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 여수웅천', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트 대천파로스', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데리조트 부여', '롯데리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('토모노야 호텔&료칸 대천점', '토모노야호텔앤료칸')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('디바인호텔 광양점', '디바인호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다프라자 바이윈덤 여수', '라마다프라자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('데이나이스호텔 대천지점', '데이나이스호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('남원예촌by켄싱턴', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('자우리호텔 도안점', '자우리호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다호텔 대전', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더시티호텔 군산', '더시티호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아호텔 여수', '베니키아')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트 제주', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다호텔함덕', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데호텔제주', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('유탑유블레스호텔 제주', '유탑유블레스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베스트웨스턴가족제주호텔', '베스트웨스턴')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운스위트제주 호텔앤리조트', '브라운스위트제주호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주 브라운스위트 호텔', '브라운스위트제주호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코업시티호텔성산', '코업시티호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('서머셋제주신화월드', '제주신화월드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다앙코르서귀포호텔', '라마다앙코르')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데시티호텔 제주', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신신호텔 서귀포점', '신신호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('메리어트관 제주신화월드호텔앤리조트', '제주신화월드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더베스트제주성산호텔', '더베스트호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스타즈호텔 제주로베로', '스타즈호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더스위트호텔 제주', '더스위트호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('코업시티호텔하버뷰', '코업시티호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('소노벨 제주', '소노벨')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('골든튤립제주성산호텔', '골든튤립')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주신라호텔', '신라호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신라스테이 제주', '신라스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주비스타케이호텔', '비스타케이호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아루미호텔 협재점', '아루미호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주금호리조트', '금호리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 제주중문', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주부영호텔앤리조트', '부영호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데리조트제주 아트빌라스', '롯데리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('FK알레그리아호텔', '알레그리아호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('알레그리아호텔서귀포', '알레그리아호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('노마드인 제주', '노마드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('팜파스호텔 제주', '팜파스호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('팜파스호텔&리조트', '팜파스호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('오션팰리스호텔제주', '오션팰리스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주오션팰리스', '오션팰리스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다호텔제주', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주지방해양경찰청 수련원', '해양경찰교육원')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('파르나스호텔 제주', '파르나스호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('켄싱턴리조트 서귀포', '켄싱턴리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔스카이파크 제주1호점', '호텔스카이파크')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('썬라이즈중문호텔', '썬라이즈')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('썬라이즈호텔 성산', '썬라이즈')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스위트오션 스위트메이서귀포호텔', '스위트오션')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화리조트용인', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('랜딩관 제주신화월드 호텔앤리조트', '제주신화월드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('휘닉스제주', '휘닉스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('뚜르드제주게스트하우스', '뚜르드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('뚜르드포항', '뚜르드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스테이인터뷰 금산', '스테이인터뷰')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스테이인터뷰 대전', '스테이인터뷰')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스테이인터뷰발라드우', '스테이인터뷰')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔 스테이인터뷰 제주', '스테이인터뷰')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('그랜드하얏트제주', '그랜드하얏트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('하이원리조트 삼척 삼척', '하이원리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신신호텔 제주공항', '신신호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신신호텔 제주월드컵', '신신호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('신신호텔 천지연', '신신호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('서귀포JS호텔', 'JS호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('한화호텔 용인', '한화리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주갤럭시호텔', '갤럭시호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('갤럭시 모텔', '갤럭시호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('에벤에셀함덕호텔', '에벤에셀호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주아이브리조트', '아이브리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('제주명성아카데미하우스', '명성아카데미하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('블랙스톤리조트 제주', '블랙스톤')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('블랙스톤벨포레리조트', '블랙스톤')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('여수유탑유블레스', '유탑유블레스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('저스트슬립호텔 당진터미널점', '저스트슬립')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('포레스트리솜', '리솜')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 제주연동', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('어반스테이 인천송도', '어반스테이')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('천안상록호텔', '상록호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('JNK클래식호텔 그레이 2호점', 'JNK호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔케니 여수', '호텔케니')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 광주역점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔야자 광주첨단점', '호텔야자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('M부띠끄모텔', 'M모텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('창원M모텔', 'M모텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('쉐라톤비지니스텔', '쉐라톤')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다군산호텔', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다호텔 함덕', '라마다호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('천안 소노벨', '소노벨')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더원무인호텔', '호텔더원')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('더리버사이드호텔', '리버사이드호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('모텔피아노', '피아노모텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 아산터미널점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트호텔 순천역점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 부안변산점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 삼천포항점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('브라운도트 제천역점', '브라운도트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('여수베네치아호텔앤리조트', '베네치아호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('레스트리리솜', '리솜')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('아일랜드 리솜', '리솜')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('탑클라우드호텔 익산', '탑클라우드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('탑클라우드 군산', '탑클라우드')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아중문 호텔', '베니키아')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아관광호텔', '베니키아')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('베니키아호텔대림', '베니키아')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다프라자바이윈덤 여수', '라마다프라자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('라마다프라자호텔&씨원리조트', '라마다프라자')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('24게스트하우스 여수점', '24게스트하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('순천24게스트하우스', '24게스트하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('스플라스 리솜', '리솜')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('하이원리조트 마운틴콘도', '하이원리조트')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('호텔자우리', '자우리호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('자우리호텔 당진점', '자우리호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('롯데호텔 서울', '롯데호텔')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('103LAB 게스트하우스', '103랩')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('힐하우스 대전', '힐하우스')
df['VISIT_AREA_NM'] = df['VISIT_AREA_NM'].replace('힐하우스 보령', '힐하우스')

fdf = df.copy()

fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마루호텔 본관','마루호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('페어필드 바이메리어트 서울','페어필드바이메리어트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나무늘보호텔 순천연향점','나무늘보호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔여기어때 광주역점','호텔여기어때')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 광주하남점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑클라우드호텔 군산점','탑클라우드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('뜰식당','뜰펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('저스트슬립호텔 유성온천점','저스트슬립호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코트야드바이 메리어트 세종','코트야드바이메리어트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블랙스톤 벨포레리조트 웰컴센터','벨포레리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블랙스톤','벨포레리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 공주신관점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('순천24게스트하우스2순천24게스트하우스4순천24게스트하우스게순천24게스트하우스스순천24게스트하우스트순천24게스트하우스하순천24게스트하우스우순천24게스트하우스스순천24게스트하우스 순천24게스트하우스인순천24게스트하우스호순천24게스트하우스텔순천24게스트하우스','순천24게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 충주역점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베스트웨스턴 플러스호텔 세종점','베스트웨스턴플러스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 아늑호텔 천안역2호점','더휴식아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베스트웨스턴호텔 군산','베스트웨스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 봄','호텔봄')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('케이스부띠끄호텔 제천점','케이스부띠끄호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제천시 청풍면 학현소야로 473(제천경찰수련원)','제천경찰수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔원 변산점','호텔원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 여수학동점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔그레이톤 둔산','호텔그레이톤')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('좌구산자연휴양림 별무리동','좌구산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('T스테이미니호텔 청주점','T스테이미니호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봉수산자연휴양림 은방울/참나리동','봉수산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('자연드림 괴산자연드림파크점','자연드림파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반스테이 여수오션힐','어반스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자호텔 자은도','라마다프라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레이크힐스 속리산호텔','레이크힐스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인호텔 정부청사앞점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔G7 군산4호점','호텔G7')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베스트웨스턴호텔 군산','베스트웨스턴호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑클라우드호텔 천안점','탑클라우드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔디바인 순천점','호텔디바인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25모텔','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코드야드 바이 메리어트 세종','코드야드바이메리어트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트 충주','켄싱턴리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청풍리조트 힐하우스','청풍리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('글로스터호텔 청주','글로스터호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스테이인터뷰 태안','스테이인터뷰')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전북 김제시 금산면 우림로 256-6 1층 및 2층','펜션마마')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('전북 김제시 금산면 우림로 256-6','펜션마마')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('만월호텔 유성점','만월호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('만월호텔 대흥점','만월호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 호텔 (천안역점)','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오헤브데이호텔 남원지점','오헤브데이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('괴산자연드림파크 호텔로움','호텔로움')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 아늑호텔 유성2호점','더휴식아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인호텔 정부청사앞점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔원 변산점','호텔원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('저스트스테이 호텔 공주신관점','저스트스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이본호텔 군산','에이본호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('만월호텔 중리점','만월호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스테리움 제천','스테리움')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('공주호텔INK','공주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 대천파로스 (체크인 5시30분경)','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이 여수','신라스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('부여군시설관리공단 유스호스텔 계백관','부여군시설관리공단유스호스텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다 앙코르 바이 윈덤 천안','라마다앙코르')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캠프 202','캠프202')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블랙스톤벨포레리조트 벨포레목장','벨포레리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자바이윈덤 충장','라마다프라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('교동살래_숙박지','교동살래')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포레스트리솜 빌라동 47동','리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('측후동19번지','측후동19번지게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다락휴 여수점','다락휴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('청주 동보원 자연 휴양림 팬션','동보원자연휴양림팬션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('모항레저타운 B동','모항레저타운')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('군산 후즈넥스트','후즈넥스트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라포레스타 제천','라포레스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('케이스부띠끄호텔 제천점','케이스부띠끄호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('자우리호텔 유성점','자우리호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('K부티크호텔','케이스부띠끄호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비토애럭셔리글램핑 나주점','비토애럭셔리글램핑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 순천역점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('성주산자연휴양림 물놀이장','성주산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔마스타대천','호텔마스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠에이치호텔 객리단길점','엠에이치호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오헤브데이호텔 남원지점','오헤브데이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔마루 단양','호텔마루')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포레스트리솜 해브나인스파','리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코업스테이 코아루여수 1층 조식식당','코업스테이 코아루여수')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('일번지브띠끄호텔 충장로점','일번지브띠끄호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무주나봄리조트 별관','무주나봄리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아이리스 숙박','아이리스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('노두갤러리','1004민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다플라자 광주호텔','라마다플라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베니키아 호텔대림','베니키아호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 아늑 청주우암2호점','더휴식아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 시청점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다락휴 여수점','다락휴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JI호텔 신관','JI호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('저스트스테이 호텔 아산시청점','저스트스테이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반스테이 여수오션힐','어반스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨사이드인여수','씨사이드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 로씨오 카페','호텔로씨오')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽레스피아in태안','클럽레스피아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트 충주','켄싱턴리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔브라운도트','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔디바인 순천점','호텔디바인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('엠에이치호텔 객리단길점','엠에이치호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무주덕유산리조트 가족호텔 코스모스동','무주덕유산리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔원 변산점','호텔원')
# 2178까지 완료
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대이작도 마린스포츠','대이작마린스포츠')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 아늑호텔 월곶지점','더휴식아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑클라우드호텔 수원점','탑클라우드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스타즈호텔 동탄점','스타즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인터내셔널호텔 영종','인터내셔널호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서초 제이에스 (JS) 호텔 (Seocho JS Hotel)','JS호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('천안 브라운도트 사직점','브라운토트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('힐튼가든인 서울강남점 조식 뷔페','힐튼가든인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 여기어때 안양1번가점','호텔여기어때')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 용인점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('로컬스티치 시청','로컬스티치')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('네스트호텔 쿤스트라운지','네스트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔꼬뮨 광릉수목원점','호텔꼬뮨')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크우드 프리미어 인천','오크우드프리미어')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쉐라톤그랜드 인천호텔 22층 라운지','쉐라톤그랜드 인천호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔써클 합정점','호텔써클')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('홀리데이인인천송도','홀리데이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다앙코르 평택호텔','라마다앙코르')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더위크앤 리조트','더위크앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아시아드반월호텔 구월동점','아시아드반월호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파크마린호텔수영장','파크마린호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더노벰버스테이 송도랜드마크점','더노벰버스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 을왕리점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔소프라 인천청라점','호텔소프라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드하얏트 인천','그랜드하얏트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 인천주안역점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코오롱호텔 경주','코오롱호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인호텔 울산삼산점 울산삼산점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파라다이스호텔부산 신관','파라다이스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('아바니 센트럴 부산','아바니센트럴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('웨스틴조선 부산','웨스틴조선')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 경주','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트 경주','켄싱턴리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스위트호텔 경주','스위트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리빙캐슬 부산대점','리빙캐슬')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 누리다 안동','호텔누리다')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이 여수','신라스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랩디오션 송도','그랩디오션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('와이컬렉션 by UH FLAT 오시리아','마티에 오시리아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리센오션파크속초A동','리센오션파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 범일점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('리센오션파크속초B동','리센오션파크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 더스테이 낭만','호텔더스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스카이베이호텔 경포','스카이베이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 포항여객터미널점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('연화문호텔 영덕','연화문호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('원웨이 부산점','원웨이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다호텔 평창','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 진하해수욕장점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('크라운하버호텔 부산','크라운하버호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이 서부산','신라스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오션투유리조트 속초설악비치 호텔앤콘도','오션투유콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 홍천IC점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('속초아이파크스위트호텔앤레지던스','아이파크콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨크루즈호텔속초','씨크루즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('월드스테이트 속초','월드스테이트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 초읍점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하운드호텔 연산','하운드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인호텔 부산중앙역점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('2월호텔 송도암남점','2월호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('헤븐마크인속초','헤븐마크')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스카이베이호텔 경포','스카이베이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봉수산자연휴양림 가야산동','봉수산자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑클라우드호텔 군산점','탑클라우드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔브라운도트 광주충장로점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반스테이 여수오션힐','어반스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 목포갓바위점','브라운도트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑클라우드호텔 익산점','탑클라우드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포레스트리솜 빌라동 30동','리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('탑스텐리조트 동강시스타','탑스텐리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('씨사이드인여수호텔앤리조트','씨사이드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다락휴 여수점','다락휴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자여수','라마다프라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔케니여수','호텔케니')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔디바인 순천점','호텔디바인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 신화관','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해비치 호텔&리조트 제주','해비치호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('애월스테이인제주','애월스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔휴식 서귀포','호텔휴식')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플레이스캠프 제주','플레이스캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 랜딩관','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휴스테이 금호','휴스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비스타케이호텔 월드컵','비스타케이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주웨스턴그레이스 강정점','제주웨스턴그레이스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플레이스캠프 제주','플레이스캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 서머셋','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔브릿지 서귀포','호텔브릿지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그라벨호텔제주','그라벨호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JW메리어트제주리조트&스파','JW메리어트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('벤티모호텔앤레지던스제주','벤티모호텔앤레지던스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다인오세아노호텔 인피니트 풀','다인오세아노호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스탠포드호텔앤리조트 제주','스탠포드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('머큐어앰배서더 제주','머큐어앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새마을금고중앙회 새마을금고제주연수원','새마을금고제주연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('샐리스 제주(수영)','샐리스제주호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하운드호텔 영광점','하운드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 휘슬락 바이 베스트웨스턴 시그니처 컬렉션','호텔휘슬락')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('함덕비치스테이제주','함덕비치스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 메리어트관','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 서머셋','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('롯데리조트제주 아트빌라스 B블럭','롯데리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대아울릉리조트 울릉지점','대아울릉리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('비스비제주 앤 팜빌리지','비스비제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 신화빌라스5단지','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에코랜드 호텔','에코랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에코랜드호텔 레스토랑','에코랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에코랜드 레스토랑','에코랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플레이스캠프 제주','플레이스캠프')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('함덕제주비치호텔','함덕비치스테이제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해비치 호텔&리조트 제주 수영장','해비치호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('무릉리3845-5','델루나스파펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주웨스턴그레이스 강정점','제주웨스턴그레이스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔골든데이지 서귀포오션','호텔골든데이지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('일성콘도앤리조트 제주비치지점','일성콘도앤리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('타마라 제주','타마라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다인오세아노호텔 루프탑ARA','다인오세아노호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔골든데이지 서귀포오션','호텔골든데이지')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬라이즈호텔 섭지코지점','썬라이즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신화워터파크','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔휴식 서귀포','호텔휴식')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신신호텔 천지연점','신신호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신신호텔 제주오션','신신호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 신화관','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 랜딩관','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 서머셋','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 메리어트관','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신화월드 신화빌라스5단지','제주신화월드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴리조트 경주','켄싱턴리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주해비치호텔앤리조트 수영장1','해비치 호텔&리조트 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('카이리조트수영장','카이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자여수','라마다프라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포레스트리솜 빌라동 30동','리솜')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주웨스턴그레이스 강정점','웨스턴그레이스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('절물자연휴양림 장생의숲길','절물자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('블룸호텔수영장','블룸호텔제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라호텔 제주','신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('다인오세아노호텔 루프탑ARA','다인오세아노호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주신라호텔수영장','신라호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('GS25 한화리조트제주점','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새마을금고연수원 본관','새마을금고연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새마을금고중앙회 새마을금고제주연수원','새마을금고연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('판포리 135','판포리135')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('그랜드조선제주 아리아','그랜드조선 제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자 제주호텔 수영장','라마다프라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더포그레스호텔앤리조트','더포그레이리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('제주 서귀포시 이어도로 738','브리지스튜디오')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스평창 유로빌라','휘닉스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('센텀 스위트 호텔','센텀스위트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 아늑호텔 부산남포점','더휴식 아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 노마드 아늑호텔 구미지점','더휴식 아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원 그랜드호텔 컨벤션타워','하이원 그랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원추추파크 네이처빌','하이원리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원리조트 힐콘도','하이원리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원팰리스호텔','하이원 그랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경남 사천시 서포면 자구로 69-30번지','y1펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄델피노 AB동','소노캄')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KT&G상상마당 부산','KT&G상상마당')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('KT&G상상마당 춘천스테이호텔','KT&G상상마당')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립산림치유원 문필마을식당','국립산림치유원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국립산림치유원 문필마을 MB101','국립산림치유원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 대구성서점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 창원명서컨벤션센터점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 김해삼계점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트 노스콘도','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리스키빌리지 수영장','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트 스키콘도D','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트 빌리지센터','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리리조트 콘도C동','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오크밸리 스키빌리지C동','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 사상르네시떼점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 남춘천점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 경주점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 부산진역점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 송도해수욕장점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 포항죽도점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 경주양남점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 정자해수욕장점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 보문점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 울산학산점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 양산석산점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 일광해수욕장점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 해운대점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 정관점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨비발디파크','소노벨')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노벨청송 솔샘온천','소노벨')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('부산 하단 그릴힐','그린힐모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25호텔 서면역점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25호텔 청초호점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25호텔 광안리점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 울산번영로점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25호텔 대구역점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 불국사점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25호텔 광안리점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('삼척 넘버25 삼척시청점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('신라스테이 서부산','신라스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('봉좌마을교류센터','봉좌마을캠핑장')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노펠리체 델피노 더엠브로시아','소노펠리체 델피노')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노펠리체 빌리지비발디파크 O동','소노펠리체 델피노')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노펠리체 빌리지 델피노','소노펠리체 델피노')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하와이 호텔','하와이호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('체스터톤스 속초','체스터톤스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오투리조트 타워콘도','오투리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H에비뉴 광안리해변점','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H에비뉴 기장일광점','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H에비뉴호텔 광안리점','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H에비뉴 동성로 대구역점','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('르네상스 호텔','르네상스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄 델피노','소노캄')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('속초 체스터톤즈','체스터톤스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다호텔 평창','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다강원태백호텔','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H avenue hotel','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('휘닉스평창 오렌지동','휘닉스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이치애비뉴 기장 일광점','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이치 애비뉴','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('레고랜드코리아리조트 레고랜드호텔','레고랜드코리아리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프룩스플럭스호텔수영장','프룩스플럭스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('프룩스플럭스호텔루프탑','프룩스플럭스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해링턴타워광안디오션오피스텔','해링턴하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인호텔 부산중앙역점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인 대구동성로점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알펜시아리조트 (가족 회사 복지혜택으로 숙박 지원)','알펜시아리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('고성금강산콘도 바베큐장','고성금강산콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 경주','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 백암온천 온천사우나','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 백암온천','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('E7양양죽도 오션스테이양양 1208호','오션스테이양양')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코오롱호텔 경주 사우나','코오롱호텔 경주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('h avenue hotel','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('강원특별자치도교육청교직원수련원 아라리분원','교육청교직원수련원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('UH suite the Haeundae','UH스위트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('uh suite the gyeongju','UH스위트호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('평창군 봉평면 태기로 569','위 모텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄델피노 AB동','소노캄')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소노캄 델피노','소노캄')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('H avenue','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('Havenue','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파크로쉬리조트앤웰니스 글라스하우스','파크로쉬리조트앤웰니스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경북 경주시 외동읍 토함산로 41-4','리포소풀빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경북 경주시 태종로685번길 20(에이치애비뉴 호텔)','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스스타일앰배서더 명동','이비스앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스앰배서더호텔 부산시티센터점','이비스앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하이원그랜드호텔','하이원 그랜드호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알펜시아 홀리데이인리조트호텔','알펜시아리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('인터컨티넨탈 알펜시아리조트','알펜시아리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('알펜시아 인터컨티넨탈호텔','알펜시아리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('속초IPARK스위트','속초아이파크스위트호텔앤레지던스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주 게스트하우스 프렌즈','경주게스트하우스 프렌즈')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베스트웨스턴 해운대호텔','베스트웨스턴플러스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('루오스테이 수월','루오')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라한호텔','라한')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에이치에비뉴 호텔','에이치에비뉴')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔여기어때 강릉경포점','호텔여기어때')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔 여기어때 경주보문점','호텔 여기어때')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔5월','오월')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 용인베잔송','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('한화리조트 산정호수안시','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인호텔 서울강남점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('토요코인 대구동성로점','토요코인호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동두천자연휴양림 복층4호','동두천자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반스테이 부티크 익선','어반스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 시흥정왕점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 정왕점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트호텔 월곶점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('브라운도트 용인점','브라운도트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25호텔 신촌점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔넘버25 동수원사거리점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('넘버25 사당역점','넘버25')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쉐라톤그랜드인천호텔 피스트','쉐라톤그랜드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('포포인츠바이쉐라톤 수원호텔','포포인츠바이쉐라톤')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코트야드바이메리어트 서울남대문','코트야드바이메리어트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경기 김포시 김포한강9로 80 다온프라자 7층','캘리포니아 호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마이다스호텔&리조트 로비','마이다스호텔&리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('마이다스호텔&리조트 룸','마이다스호텔&리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나인트리 프리미어 로카우스 호텔 서울 용산','나인트리프리미어호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('스타즈호텔 독산 체크인','스타즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나랑','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('에덴파라다이스호텔근처  산책로','에덴파라다이스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스스타일앰배서더 인천에어포트','이비스앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스스타일앰배서더 서울용산','이비스앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스앰배서더호텔 수원','이비스앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('이비스앰배서더 서울강남','이비스앰배서더')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('곤지암리조트 West빌리지','곤지암리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('곤지암리조트 L빌리지','곤지암리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('곤지암리조트 East빌리지','곤지암리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 구리수택점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔야자 오산역점','호텔야자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('플라자CC 용인','한화리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('솔마루파인팜하우스 3호','솔마루파인팜하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('the 조선호텔 수안보','더조선호텔 수안보')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔더루아 충장로점','호텔더루아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('깜찌기 스테이','유탑유블레스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('The 조선호텔 수안보','더조선호텔 수안보')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라온프라이빗타운','라온호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('소설스미스호텔 천안점','소설스미스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('구공스테이 서산 보담','구공스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('동보원','동보원자연휴양림팬션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('더휴식 아늑호텔 천안역 1호점','더휴식아늑호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('클럽레스피아in 태안','클럽레스피아')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('구공스테이안온','구공스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('위코스테이 영종','위코스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주 하녹풀빌라','하녹풀빌라')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('남해 온뷰','온뷰펜션')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('대천어썸게스트하우스','어썸게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('달 게스트하우스','달게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다프라자&씨원리조트 자은도','라마다프라자')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('나주중흥골드스파&리조트 골드동','중흥골드스파&리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('해비치 호텔','해비치호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('함덕비치스테이제주','함덕비치스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔위드제주','호텔위드')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('JW메리어트 제주 연회장','JW메리어트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오브젝트 제주선흘점','오브젝트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('시드니','시드니호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오케이코퍼레이션(서귀포 JS호텔)','JS호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('국군복지단 대천콘도마트','국군복지단 대천콘도')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('모슬포호텔 3동','모슬포호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔더그랑 중문','호텔더그랑')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('어반스테이 명동','어반스테이')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('하워드존슨호텔 인천에어포트','하워드존슨호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오피스제주 사계점','오피스제주')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새마을금고 제주연수원','새마을금고연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('새마을금고제주연수원','새마을금고연수원')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베드라디오 동문점','베드라디오')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('오투힐 제주점','오투힐')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔통 연동점','호텔통')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('썬라이즈 호텔 성산점','썬라이즈호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('쉼표더하기게스트하우스 여수점','쉼표더하기게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔화인 제주','호텔화인')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('파크힐호텔 완도','파크힐호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('베네치아휴안','스테이휴안')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서귀포자연휴양림 담팔수동','서귀포자연휴양림')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('온더스톤게스트하우스 2호점','온더스톤게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('설레임게스트하우스 2호점','설레임게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('캄/kalm','kalm민박')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('설레임게스트하우스 애월 1호점','설레임게스트하우스')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('코오롱호텔 경주','코오롱호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타 테마 키즈호텔','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타호텔 화명점','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타포레스트호텔','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타포레스트','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타 마산어시장점','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('덴바스타호텔 헤리티지 경주점','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('사직덴바스타 시그니처호텔','덴바스타호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('경주 덴바스타 호텔 헤리티지','덴바스타')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('켄싱턴호텔 평창 글램핑빌리지','켄싱턴리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('뮤지엄산','오크밸리리조트')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다 서울동대문','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다바이원덤 서울동대문','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다호텔앤스위트 서울남대문','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('라마다 평택호텔','라마다호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('호텔헤르메스','헤르메스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('헤르메스호텔 에버랜드점','헤르메스호텔')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울큐브이화','서울큐브')
fdf['VISIT_AREA_NM'] = fdf['VISIT_AREA_NM'].replace('서울큐브 이태원','서울큐브')




# 의미 없는 변수 제거
words_to_remove = ['방파제','D&A빌라', '산들강웅포마을', '공주 연미산 자연공원 관광 스테이', '장안창작마당','제2연화봉 대피소',
                  '목포 대중음악의 전당', '월성산마을회관', '고목정', '옥천전통문화체험관', '용머리고을','단밤',
                  '궁남지', '아산외암리참판댁', '일락당사랑채', '유월드', '왕포마을복지관', '왕포마을  복지관', '왕포마을 복지관',
                  '진안고원치유숲', '옥천 농막', '동반자 본가', '배바우 도농교류센터', '동반자본가', '투썸플레이스 소노벨천안점',
                  '지리산생태체험단지', '옥천전통문화체험관','금성연립', '운여해변', '증평 리모텔', '베테랑','청호수마을',
                  '충청북도괴산증평교육지원청 쌍곡휴양소', '고창 웰파크시티 힐링카운티', '복대동 13-7','충남 아산시 도고면 기곡리',
                  '주인테이블', '자연드림 괴산자연드림파크점', '괴산자연드림파크', '시화연풍', '문당환경농업마을',
                  '좋은하루1차', '춘향가', '오성한옥문화센타', '전동성당', '신리로225', '줄포상설시장', '옥천전통문화체험관',
                  '제일스포츠', '국민건강보험공단 인재개발원', '신양1길98-6', '무등산생태탐방원', '가연당', '마이산토담',
                  '외암민속마을', 'Part.1', '이가신정길77', '한빛빌라', 'CU 부안고사포점','그랑메종', '중산낚시터',
                  '효문화마을관리원', '보성녹차밭','국립오서산자연휴양림', '고창읍성한옥마을', '셀픽스', '정안알밤휴게소 논산방향',
                  '토담마실', '통큰해물뚝배기삼합', '무풍승지체험마을', '춘향가', '올드브릭스', '멀꼼하우스', '군산시외버스터미널',
                  '상무e-다움골드(상무이다움골드)', '장태산자연휴양림 산림문화휴양관', '더허브오피스텔', '변산반도생태탐방원',
                  '목포 갓바위', '왕새우직판장', '홀리데이인 익스프레스']

# 지정된 단어가 포함된 행을 제거함
fdf = fdf[~fdf['VISIT_AREA_NM'].str.contains('|'.join(words_to_remove))]

# 의미 없는 변수 제거
words_to_remove1 = ['HB엘림캐슬오피스텔', '밧지름해변', '서포리해변','광백저수지', '뚝섬한강공원','라이온스쉼터',
                  'Haeandong-ro 전원주택, Seonwon-myeon, Ganghwa-gun, 인천 417-820', '하동진','일루아펜트하우스',
                  '양평군 청운면 삼성리', '종로 35번길 10', '실미도유원지', '코지네이쳐농업회사법인', '산장관광지',
                  '뒷장술해수욕장','대부북동 1622', '하나개해수욕장 방갈로', '이마트24 영종덕교중앙점', '함백산돌솥밥',
                  '파아란지붕', '수요리성', '대구공항', 'BLANC', '속초더블루테라오피스텔', '시골살이농촌체험휴양마을',
                  '나무늘보, 요정(전북 완주군 소양면 위봉길33)', '만흥동 19-1 충장휴양소', '동작구휴양소',
                  '앞섬체험센터', '원촌','보은 삼년산성', '압록유원지', '여수우정교육센터', 'G340', '구르미스시&카페', '모슬포항등대',
                  '사진놀이터', '김포공항', '곁겹', '재주에서돌고래를만난다면','고기역', '하동별관', 'CU 편의점',
                  '서귀포자연휴양림', '냥꼬게스트하우스', '제주 스위스마을 304동', '청춘당찹쌀꽈배기 동부점', '킹덤PC방',
                  '삼양사 제주휴양소', '야크마을', '델문도', '삼다헌', '함덕천마에버하임1차', '김영관센터', '초풍',
                  '제주서핑스쿨', '삼다수숲길', '바다위의언덕', '롯데 아트빌라스', '고양이수염', '바나나보드카페',
                  '거제도학동스테이새록','동광리 강낭오름 근처', '영종씨사이드 레일바이크','청운고시원','종달리수국길',
                   '조천읍 신촌북3길 131','백패킹 박지 형제해안로 잔디','금능해수욕장','명가솥뚜껑',
                   '광섭이네낚시좌대','순천만국가정원 그랜드조선가든스테이','국립공원공단 내장산생태탐방원',
                   '명품전통육개장','제주화북동우정교육센터', '울릉크루즈투어', '로비나발리168',
                   '수목원길야시장','','제주화북동우정교육센터', '롯데ATM 세븐일레븐 서귀포더포그레(CD)',
                   '한림읍 동명리 1466-1','서귀포매일올레시장', '오는정김밥', '제주공항', 'PUB섬노예',
                   '강화 황산도 해변로', '끝섬 전망대', '여수엑스포역', '하도서길 4-15 이투스 공감동',
                   '울릉크루즈', '목포국제터미널', '연화도 전망대', '중문색달해수욕장', '정가네연탄구이',
                   '곰막식당', '서귀포매일올레시장','대박명가b', '꼬지사께 제주모슬포점', '제주서핑스쿨','중리어촌체험마을',
                    '국립변산자연휴양림', '미륵산골다목적센터', '구례자연드림파크','힐링캠프트리팜', '새별공원',
                   '호우섬 신세계아트앤사이언스점', '나라매점', '인천좌대낚시', '로그더스페이스','스테이읻다',
                   '아이리스', '정우빌라트', '천호주택', '테이블730', '유성푸르지오시티', '상무e-다움골드(상무이다움골드)',
                   '오형제좌대', '동자북문화역사마을','Haeandong-ro 전원주택',  '강화도래미마을', '간절곶등대',
                   '아바니센트럴', '거제식물원 정글돔', '상화원 한국빌라','선봉스포렉스', '행담도휴게소 양방향', '신덕해수욕장',
                   '한세월파크식당', '중앙빌', '마이온홈', '곽지해수욕장', '빛의 벙커','무한의다리', '스누즈',
                   '오션팰리스', '미도스카이빌', '해일월', '도민상회 월정점', '토마토빌리지', '와인점방', '폴개협동조합',
                   '엠제이벤처오름오피스텔', '다산네트웍스 제주연구원', '선한종이', '제주1번가더테라스오피스텔','Coda',
                   '청목더웰연동', '주연', '방죽포해수욕장', '하도서길4-15 이투스 공감동', '하도서길 4-15이투스 공감동',
                   '제주목 관아', '한림 회사휴양소', '함덕8길5', '애월 담담', '서광이지', '모던테라스', '굿스프링스',
                   '죽암스테이','힐스테이트삼송역스칸센오피스텔(2블록)']

# 기존 방식에서 제거 되지 않은 변수 추가 제거
for i in range(len(words_to_remove1)):
    fdf[fdf['VISIT_AREA_NM']==words_to_remove1[i]].index
    fdf=fdf.drop(fdf[fdf['VISIT_AREA_NM']==words_to_remove1[i]].index)

words_to_remove2 = ['방파제', 'D&A빌라', '강문해변','카우치 포테이토','해운대한신휴플러스오피스텔',
                    '한덕교','합천돼지국밥', '광안KCC스위첸하버뷰', '바루서프', '전은경서프카페',
                    '소백산자락길방문자센터', '톤쇼우', '부산본가', '천도회관', '프리즈모텔',
                    '장산역 부산2호선','강릉 그림반 이야기반 하우스','계룡로6길22', '연화정',
                    '크라브텐', '송도해수욕장','더바른식당 수성점','석주재', '부흥리해수욕장',
                    '행복한빅마마삼겹 부산본점', '문막휴게소 강릉방향', '기차 안', '천안시 동남구 목천읍 교천지산길 284-6',
                    '강현면215-850','제이여성병원','유원빌라','천왕봉하누골먹돼지', '경북대학교 경북대학병원 인재원',
                    '라마다앙코르해운대호텔라라마다앙코르해운대호텔마라마다앙코르해운대호텔다라마다앙코르해운대호텔앙라마다앙코르해운대호텔코라마다앙코르해운대호텔르라마다앙코르해운대호텔해라마다앙코르해운대호텔운라마다앙코르해운대호텔대라마다앙코르해운대호텔호라마다앙코르해운대호텔텔라마다앙코르해운대호텔',
                    '한솔식자재마트','강현면215-850','세븐일레븐 속초등대점','금마늘다목적센터',
                    '템플온더비치 클럽','계곡', '경포호', '경상북도 경주시 노서동 27-2',
                    '인구해변', '산아래', '대덕산 태백시용연동굴','안반데기','CU 부산G7점',
                    '하동지리산흑돼지', '선자령 빽패킹 장소','세웅빌딩', '피아노', '대구동문동현대썬앤빌중앙로오피스텔',
                    '예다인','서울시 강북구 송정동','계양우림카이저팰리스오피스텔', '중앙선술', '동대구유성푸르나임', '명동식당', '농암종택', '성주막창 대명점', '롯데월드 어드벤처 부산',
                    '속초교동우체국'
                   ]

# Filter out rows containing any of the specified words
fdf = fdf[~fdf['VISIT_AREA_NM'].str.contains('|'.join(words_to_remove2))]

words_to_remove3 = ['봉화회관','현대프리미엄아울렛 김포점','힐스테이트에코마곡나루역오피스텔',
                    '힐스테이트에코마곡오피스텔','바나나PC','지뜨','힐스테이트삼송역스칸센오피스텔(2블록)',
                    '자연과별가평천문대', '무아레 도그라운드', '경기도 용인시 양지면 식금리 182-46',
                    '장화리낙조마을','평화누리체험장','황새울로258번길 35','경기도 양평군 단월면 석산리 산 100-4',
                    '동교로','싱싱하우스협동조합','마전낚시터','고색리치아노 오피스텔','지수물길86',
                    '레즈고스터디카페 양평점','에버랜드 T익스프레스 와 아마존','영흥바다낚시터','토실이네','la casa  (라-까사)',
                    '고덕역효성해링턴타워더퍼스트오피스텔','큰삼촌농촌체험여행','취옹예술관', '깊은산계곡산장',
                    '너리굴문화마을','강가네떡볶이','대궐숯불장어촌','산음리 211','자연다슬기해장국','보영만두 북문본점',
                    '구리갈매대방디엠시티메트로카운티오피스텔','르컬렉티브 시흥 웨이브파크','청파책가도','망능리43-7',
                    '파라스파라 서울 104동','르컬렉티브 시흥 웨이브파크', '경기 가평군 상면 임초밤안골로 205',
                    '건대입구역자이엘라오피스텔', '사나사','경기도 양평군 용문면 상망길 143-7',
                    '광성오피스텔','조치원역인근', '산정호수', '나만의, 온도'
                   ]

# Filter out rows containing any of the specified words
fdf = fdf[~fdf['VISIT_AREA_NM'].str.contains('|'.join(words_to_remove3))]

fdf.to_csv('./2.preprocessed/pre_jeju.csv', encoding='cp949')

dfj = pd.read_csv('./2.preprocessed/pre_jeju.csv', encoding='cp949')

# 전처리에 필요한 데이터만 가져와 특정 변수에 저장
dfj = dfj[['TRAVEL_ID', 'VISIT_AREA_NM','DGSTFN']]

dfj.columns = ['userID', 'itemID', 'rating']

# 결측치 확인 후 제거
dfj.isna().value_counts()
dfj2=dfj.dropna(axis=0, how='any')

print(f'전체 데이터셋 수 : {dfj2.shape}')

# Surprise을 이용한 추천시스템
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(dfj2[['userID', 'itemID', 'rating']], reader)

# 교차검증
algo = SVD(n_factors=50, n_epochs=100)
cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=10, verbose=True, n_jobs=-1)

# 하이퍼 파라미터 튜닝
# - 코드 실행 시 튜닝 값이 미세하게 바뀔 수 있음.
algo = SVD()
param_grid = {'n_factors':[50,100,150,200],'n_epochs': [10,50,100,150], 'lr_all': [0.01,0.05,0.1], 'reg_all':[0.01,0.05,0.1]}
grid = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, joblib_verbose= 10)
grid.fit(data=data)

n_factors=list(grid.best_params['rmse'].values())[0]
n_epochs=list(grid.best_params['rmse'].values())[1]
lr_all=list(grid.best_params['rmse'].values())[2]
reg_all=list(grid.best_params['rmse'].values())[3]

print(grid.best_score['rmse'])
print(grid.best_params['rmse'])

# recall@5
def precision_recall_at_k(model, k = 5, threshold = 4):

    # 각 유저에게 predictions을 매핑.
    user_est_true = defaultdict(list)
    
    # 테스트 데이터에 대한 예측 수행
    predictions=model.test(testset)
    
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # 사용자 평가를 예측값에 따라 정렬
        user_ratings.sort(key = lambda x : x[0], reverse = True)

        # 관련된 아이템 수
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # 상위 k개 아이템 중 추천된 아이템의 수
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[ : k])

        # 상위 k개 아이템 중 관련 아이템과 추천된 아이템 수 
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >=threshold))
                              for (est, true_r) in user_ratings[ : k])

        # Precision@K: 관련 항목 중에서 추천된 항목의 비율
        # n_rec_k 이 0일 때, Precision은 정의되지않음.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: 추천된 항목 중 관련 항목의 비율
        # n_rec_k 이 0일 때, Recall은 정의되지않음.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    

    precision = round((sum(prec for prec in precisions.values()) / len(precisions)), 3)
    recall = round((sum(rec for rec in recalls.values()) / len(recalls)), 3)
    accuracy.rmse(predictions)


    print('Precision: ', precision)

    print('Recall: ', recall)
    
    # F1점수 구하는 법
    print('F_1 score: ', round((2 * precision * recall) / (precision + recall), 3))

# 훈련데이터, 테스트 데이터로 나누기
trainset, testset = train_test_split(data, test_size= 0.2, random_state = 42) 
# 도출된 최적 하이퍼파라미터 값으로 수정하여 진행
algo=SVD(n_factors = n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
predictions=algo.fit(trainset)
precision_recall_at_k(predictions, k=5, threshold=4)

# 모델 저장
with open('./4.모델저장/model/latent_model_jeju.pkl','wb') as fw:
    pickle.dump(predictions, fw)


# 최종결과값 도출과정

dataset = dfj2.values.tolist()

predictions = algo.test(dataset)

# 예측 결과를 데이터프레임으로 변환
result_dfj = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'actual_rating', 'predicted_rating', 'details'])

# 필요한 컬럼만 선택 (user_id, item_id, actual_rating, predicted_rating)
result_dfj = result_dfj[['user_id', 'item_id', 'actual_rating', 'predicted_rating']]

dfj_pivot=result_dfj.pivot_table('predicted_rating', index='user_id', columns='item_id').fillna(0)

user_item_matrix = dfj2.pivot_table(index = 'userID', columns = 'itemID', values = 'rating').fillna(0)

# 이미 방문한 숙박 업소 제외하고 추천
recommendations = []
for idx, user in enumerate(user_item_matrix.index):
    # 해당 사용자가 방문한 숙박 업소
    applied_accs = set(user_item_matrix.loc[user][user_item_matrix.loc[user] != 0].index)

    sorted_acc_indices = dfj_pivot.iloc[idx].argsort()[::-1]
    recommended_accs = [acc for acc in user_item_matrix.columns[sorted_acc_indices] if acc not in applied_accs][:5]

    for acc in recommended_accs:
        recommendations.append([user, acc])

print('로그 출력 내용이 길어 csv파일로 저장')

# csv 파일로 저장
top_recommendations = pd.DataFrame(recommendations, columns=['userID', 'itemID'])
top_recommendations.to_csv('./4.SaveModel/result/result_jeju.csv', index=False, encoding='cp949')

endtime = time.time()
X=round((endtime-starttime), 2)
print(f"실행시간 : {X}초")





