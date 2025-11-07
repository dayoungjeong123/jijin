import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="AI 쓰나미 예측 & 대응", layout="wide")
st.title("AI로 쓰나미 예측하고, 행동으로 이어가기")

# 모델 로드
@st.cache_resource
def load_model():
    return joblib.load("rf.pkl")
model = load_model()

# 데이터 입력
tab1, tab2, tab3 = st.tabs(["예측", "대응책", "데이터·한계"])

with tab1:
    st.subheader("데이터 업로드 / 샘플 사용")
    up = st.file_uploader("CSV 업로드", type="csv")
    if up:
        df = pd.read_csv(up)
    else:
        df = pd.read_csv("sample_quake.csv")  # 미리 준비
    st.dataframe(df.head())

    features = ["magnitude","depth","lat","lon","distance_to_coast"]  # 수업에 맞게 고정
    proba = model.predict_proba(df[features])[:,1]
    df["tsunami_prob"] = proba
    st.metric("평균 발생확률(%)", f"{df['tsunami_prob'].mean()*100:.1f}")
    st.dataframe(df.sort_values("tsunami_prob", ascending=False).head(20))

with tab2:
    st.subheader("예측 결과 기반 대피 가이드")
    idx = st.number_input("행 선택(인덱스)", min_value=0, max_value=len(df)-1, value=0, step=1)
    row = df.iloc[int(idx)]
    st.write("선택 지점 요약:", row[["lat","lon","magnitude","depth","tsunami_prob"]])

    # Gemini를 쓸 때는 API 호출 코드 삽입(수업에서는 프롬프트만 시연 가능)
    prompt = f"""
    위치(lat={row['lat']}, lon={row['lon']}), 규모={row['magnitude']}, 깊이={row['depth']}, 
    쓰나미확률={row['tsunami_prob']:.2f}. 
    학생과 지역주민을 위한 3단계 쓰나미 대피 가이드(즉시/단기/복구)와 체크리스트를 6문장 내로.
    """
    st.text_area("가이드 초안(프롬프트 기반)", prompt, height=150)
    st.info("※ 수업에선 Gemini 호출 대신 초안 문구를 팀별로 다듬어 최종 본문으로 사용")

with tab3:
    st.subheader("데이터·한계·출처")
    st.markdown("""
- **출처:** USGS Earthquake Catalog, NOAA Tsunami DB  
- **한계:** 내륙/지형/해저지형/실시간 관측 미반영, 오경보 가능성  
- **용도:** 학습·시뮬레이션, 정책/경보 '보조' 도구
""")
