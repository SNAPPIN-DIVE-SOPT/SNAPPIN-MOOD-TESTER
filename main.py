import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pandas as pd

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="사진 무드 분석기",
    page_icon="🎞️",
    layout="wide"
)

# ── 기본 무드 정의 ───────────────────────────────────────
DEFAULT_MOODS = [
    ("내추럴",   "candid snapshot, person not posing for camera, looking away or unaware of camera, unposed moment, walking or mid-motion, off-center framing, asymmetrical composition, natural daylight, documentary photography"),
    ("연출된",   "posed portrait, direct eye contact with camera, intentional pose, fixed posture, centered subject, symmetrical composition, clearly staged scene, deliberate setup"),
    ("서사적",   "person interacting with objects, action in progress, contextual background, storytelling scene, cinematic still, narrative moment"),
    ("따스한",   "warm sunlight, golden hour, sunset light, yellow-orange tone, soft shadows, natural warm light"),
    ("청량한",   "bright daylight, clear blue sky, vivid green nature, outdoor scene, open space, fresh air feeling, high visibility"),
    ("투명한",   "very bright exposure, white or pale background, extremely low saturation, flat lighting, almost no shadow, washed-out colors, airy and clean look"),
    ("몽환적인", "dreamy atmosphere, strong backlight, light flare, glow effect, soft focus, low contrast, ethereal mood"),
    ("뚜렷한",   "sharp focus, strong contrast, clear subject separation, defined edges, crisp details, strong visual center"),
    ("차가운",   "cool tone, blue light, night scene, artificial lighting, neon or fluorescent light, strong shadows, urban night mood"),
    ("디지털",   "digital photography, ultra sharp focus, high clarity, precise edges, clean texture, no grain, realistic color reproduction, modern camera look"),
    ("아날로그", "film photography, visible film grain, soft focus, low clarity, muted or faded colors, uneven exposure, nostalgic film look, imperfect texture"),
    ("Y2K",      "direct flash photography, point-and-shoot camera style, harsh flash shadows, overexposed highlights, strong color contrast, early 2000s snapshot aesthetic, kitschy vibe"),
]

# ── 세션 상태 초기화 ─────────────────────────────────────
if "moods" not in st.session_state:
    st.session_state.moods = [list(m) for m in DEFAULT_MOODS]
if "results" not in st.session_state:
    st.session_state.results = None
if "result_images" not in st.session_state:
    st.session_state.result_images = []
if "detail_page" not in st.session_state:
    st.session_state.detail_page = 0

# ── 모델 로드 ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

# ── 분석 함수 ─────────────────────────────────────────────
def analyze_image(img, model, tag_labels, tag_texts):
    tag_embeddings = model.encode(tag_texts)
    img_emb = model.encode(img)
    scores  = util.cos_sim(img_emb, tag_embeddings)[0].numpy()
    ranked  = sorted(zip(tag_labels, scores), key=lambda x: x[1], reverse=True)
    return ranked

# ── 사이드바: 무드 편집 ───────────────────────────────────
with st.sidebar:
    st.header("🎨 무드 설정")
    st.caption("무드 이름과 영어 설명을 자유롭게 수정하세요.")

    if st.button("➕ 무드 추가", width='stretch'):
        st.session_state.moods.append(["새 무드", "describe the mood in English"])

    st.divider()

    to_delete = None
    for i, mood in enumerate(st.session_state.moods):
        with st.expander(f"#{mood[0]}", expanded=False):
            new_name = st.text_input("무드 이름", value=mood[0], key=f"name_{i}")
            new_desc = st.text_area("영어 설명", value=mood[1], key=f"desc_{i}", height=100)
            st.session_state.moods[i] = [new_name, new_desc]

            if st.button("🗑️ 삭제", key=f"del_{i}"):
                to_delete = i

    if to_delete is not None:
        st.session_state.moods.pop(to_delete)
        st.rerun()

    st.divider()
    if st.button("↩️ 기본값으로 초기화", width='stretch'):
        st.session_state.moods = [list(m) for m in DEFAULT_MOODS]
        st.rerun()

# ── 메인: 분석 ───────────────────────────────────────────
st.title("🎞️ 사진 무드 분석기")
st.caption("사진을 올리면 왼쪽에서 설정한 무드 기준으로 자동 분석해드려요.")

tag_labels = [m[0] for m in st.session_state.moods]
tag_texts  = [m[1] for m in st.session_state.moods]

uploaded_files = st.file_uploader(
    "사진을 여러 장 한꺼번에 올려도 돼요 (JPG / PNG)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.divider()
    if st.button("✨ 분석 시작", type="primary", width='stretch'):

        with st.spinner("모델 불러오는 중... (처음 한 번만 오래 걸려요)"):
            model = load_model()

        all_rows    = []
        result_imgs = []
        progress    = st.progress(0, text="분석 중...")

        for i, file in enumerate(uploaded_files):
            img    = Image.open(file).convert("RGB")
            ranked = analyze_image(img, model, tag_labels, tag_texts)
            top3   = ranked[:3]

            all_rows.append({
                "파일명":   file.name,
                "1위 무드": f"#{top3[0][0]} ({top3[0][1]*100:.1f}%)",
                "2위 무드": f"#{top3[1][0]} ({top3[1][1]*100:.1f}%)",
                "3위 무드": f"#{top3[2][0]} ({top3[2][1]*100:.1f}%)",
            })
            result_imgs.append((file.name, img, ranked))

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"분석 중... {i+1}/{len(uploaded_files)}장"
            )

        progress.empty()

        # 결과를 session state에 저장
        st.session_state.results       = pd.DataFrame(all_rows)
        st.session_state.result_images = result_imgs
        st.session_state.detail_page   = 0

# 결과가 있으면 항상 표시 (페이지 버튼 눌러도 유지됨)
if st.session_state.results is not None:
    df = st.session_state.results

    st.success(f"✅ {len(df)}장 분석 완료!")
    st.dataframe(df, width='stretch', hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 결과 CSV 다운로드",
        data=csv,
        file_name="mood_analysis.csv",
        mime="text/csv",
        width='stretch',
    )

    st.divider()
    st.subheader("사진별 상세 결과")

    PAGE_SIZE   = 10
    total       = len(st.session_state.result_images)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    start      = st.session_state.detail_page * PAGE_SIZE
    end        = min(start + PAGE_SIZE, total)
    page_items = st.session_state.result_images[start:end]

    cols = st.columns(3)
    for i, (fname, img, ranked) in enumerate(page_items):
        with cols[i % 3]:
            st.image(img, caption=fname, width='stretch')
            for tag, score in ranked[:3]:
                st.write(f"#{tag} — {score*100:.1f}%")
            st.divider()

    # 페이지 네비게이션
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ 이전", disabled=(st.session_state.detail_page == 0), width='stretch'):
            st.session_state.detail_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px'>{st.session_state.detail_page + 1} / {total_pages} 페이지 ({start+1}–{end} / {total}장)</div>",
            unsafe_allow_html=True
        )
    with col_next:
        if st.button("다음 ▶", disabled=(st.session_state.detail_page >= total_pages - 1), width='stretch'):
            st.session_state.detail_page += 1
            st.rerun()