import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pandas as pd
import numpy as np

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="사진 무드 분석기",
    page_icon="🎞️",
    layout="wide"
)

# ── 기본 무드 정의 ───────────────────────────────────────
DEFAULT_MOODS = [
    ("내추럴", [
        "candid photo, natural expression, subject not strongly posing, relaxed posture, slight asymmetry, everyday moment, documentary feel, natural light",
        "a photo of authentic slice of life, unposed portrait, candid photography",
        "spontaneous photography, people acting naturally, unscripted moment",
    ]),
    ("따스한", [
        "warm golden hour lighting, soft sunlight, yellow-orange tones, cozy atmosphere, gentle shadows, emotionally warm and bright feeling",
        "a photo with warm tones, sun-kissed lighting, comforting and inviting mood",
        "warm color palette portrait, golden sunshine, soft and embracing light",
    ]),
    ("몽환적인", [
        "dreamy atmosphere, soft focus, glow effect, light flare, pastel tones, low contrast, hazy lighting, ethereal and surreal mood",
        "a dreamy and surreal photo, cinematic halation, magical misty lighting",
        "soft ethereal photography, gentle blur and haze, fantasy and whimsical look",
    ]),
    ("빈티지", [
        "film-like photo, visible grain, faded or muted colors, low contrast, soft clarity, analog camera feel, nostalgic tone, imperfect texture",
        "a retro vintage photograph, 35mm film grain, nostalgic color grading",
        "old school analog photography, washed out vintage colors, classic film camera aesthetic",
    ]),
    ("시크한", [
        "moody cinematic photo, cool tones, dark lighting, high contrast, dramatic shadows, sharp details, urban and modern mood",
        "a chic and moody photograph, stylish low-key lighting, cool blue shadows",
        "edgy street fashion style photo, sophisticated dark tones, modern and sleek vibe",
    ]),
    ("클린한", [
        "clean minimalist photo, high key lighting, bright exposure, white or pale background, low shadow, clear and fresh look, neat composition",
        "a visually clean and polished photo, minimalistic aesthetic, simple background",
        "bright and airy photography, pristine white balance, crisp and uncluttered view",
    ]),
]

# ── 상충 그룹 정의 (Conflict Resolution) ────────────────
# 같은 그룹 내에서 동시에 Top3에 포함되지 않도록 필터링
CONFLICT_GROUPS = [
    {"따스한", "시크한"},  # 그룹 B: 온도/톤
]

# ── 세션 상태 초기화 ─────────────────────────────────────
if "moods" not in st.session_state:
    st.session_state.moods = [[m[0], list(m[1])] for m in DEFAULT_MOODS]
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
def analyze_image(img, model, tag_labels, tag_texts_list):
    """
    tag_texts_list: List[List[str]] — 각 태그에 대한 복수 프롬프트 목록
    각 태그 점수 = 해당 태그의 모든 프롬프트 임베딩과의 코사인 유사도 평균
    """
    scores = []
    img_emb = model.encode(img)

    for texts in tag_texts_list:
        text_embeddings = model.encode(texts)
        sims = util.cos_sim(img_emb, text_embeddings)[0].numpy()
        scores.append(float(np.mean(sims)))

    ranked = sorted(zip(tag_labels, scores), key=lambda x: x[1], reverse=True)
    return ranked

# ── 상충 필터링 함수 ─────────────────────────────────────
def resolve_conflicts(ranked, top_n=3):
    """
    ranked: [(tag, score), ...] 정렬된 전체 결과
    상충 그룹 내에서 더 낮은 점수의 태그는 Top N 선정에서 제외
    """
    selected = []
    used_conflict_groups = set()  # 이미 선택된 상충 그룹 인덱스

    for tag, score in ranked:
        if len(selected) >= top_n:
            break

        # 이 태그가 속한 상충 그룹 확인
        conflict_blocked = False
        for g_idx, group in enumerate(CONFLICT_GROUPS):
            if tag in group:
                if g_idx in used_conflict_groups:
                    conflict_blocked = True
                    break

        if not conflict_blocked:
            selected.append((tag, score))
            # 해당 태그의 상충 그룹 사용 처리
            for g_idx, group in enumerate(CONFLICT_GROUPS):
                if tag in group:
                    used_conflict_groups.add(g_idx)

    return selected

# ── 사이드바: 무드 편집 ───────────────────────────────────
with st.sidebar:
    st.header("🎨 무드 설정")
    st.caption("무드 이름과 영어 프롬프트를 자유롭게 수정하세요. (줄바꿈으로 구분, 최대 3개)")

    if st.button("➕ 무드 추가", use_container_width=True):
        st.session_state.moods.append(["새 무드", ["describe the mood in English"]])

    st.divider()

    to_delete = None
    for i, mood in enumerate(st.session_state.moods):
        with st.expander(f"#{mood[0]}", expanded=False):
            new_name = st.text_input("무드 이름", value=mood[0], key=f"name_{i}")
            # 리스트 → 줄바꿈 텍스트로 표시
            prompts_text = "\n".join(mood[1]) if isinstance(mood[1], list) else mood[1]
            new_desc_raw = st.text_area("영어 프롬프트 (줄바꿈으로 구분)", value=prompts_text, key=f"desc_{i}", height=130)
            new_prompts = [line.strip() for line in new_desc_raw.splitlines() if line.strip()]
            st.session_state.moods[i] = [new_name, new_prompts]

            if st.button("🗑️ 삭제", key=f"del_{i}"):
                to_delete = i

    if to_delete is not None:
        st.session_state.moods.pop(to_delete)
        st.rerun()

    st.divider()
    if st.button("↩️ 기본값으로 초기화", use_container_width=True):
        st.session_state.moods = [[m[0], list(m[1])] for m in DEFAULT_MOODS]
        st.rerun()

# ── 메인: 분석 ───────────────────────────────────────────
st.title("🎞️ 사진 무드 분석기")
st.caption("사진을 올리면 왼쪽에서 설정한 무드 기준으로 자동 분석해드려요.")

tag_labels     = [m[0] for m in st.session_state.moods]
tag_texts_list = [m[1] if isinstance(m[1], list) else [m[1]] for m in st.session_state.moods]

uploaded_files = st.file_uploader(
    "사진을 여러 장 한꺼번에 올려도 돼요 (JPG / PNG)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.divider()
    if st.button("✨ 분석 시작", type="primary", use_container_width=True):

        with st.spinner("모델 불러오는 중... (처음 한 번만 오래 걸려요)"):
            model = load_model()

        all_rows    = []
        result_imgs = []
        progress    = st.progress(0, text="분석 중...")

        for i, file in enumerate(uploaded_files):
            img    = Image.open(file).convert("RGB")
            ranked = analyze_image(img, model, tag_labels, tag_texts_list)
            top3   = resolve_conflicts(ranked, top_n=3)

            all_rows.append({
                "파일명":   file.name,
                "1위 무드": f"#{top3[0][0]} ({top3[0][1]*100:.1f}%)" if len(top3) > 0 else "-",
                "2위 무드": f"#{top3[1][0]} ({top3[1][1]*100:.1f}%)" if len(top3) > 1 else "-",
                "3위 무드": f"#{top3[2][0]} ({top3[2][1]*100:.1f}%)" if len(top3) > 2 else "-",
            })
            result_imgs.append((file.name, img, ranked, top3))

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"분석 중... {i+1}/{len(uploaded_files)}장"
            )

        progress.empty()

        st.session_state.results       = pd.DataFrame(all_rows)
        st.session_state.result_images = result_imgs
        st.session_state.detail_page   = 0

# 결과가 있으면 항상 표시
if st.session_state.results is not None:
    df = st.session_state.results

    st.success(f"✅ {len(df)}장 분석 완료!")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 결과 CSV 다운로드",
        data=csv,
        file_name="mood_analysis.csv",
        mime="text/csv",
        use_container_width=True,
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
    for i, (fname, img, ranked, top3) in enumerate(page_items):
        with cols[i % 3]:
            st.image(img, caption=fname, use_container_width=True)
            for tag, score in top3:
                st.write(f"#{tag} — {score*100:.1f}%")
            st.divider()

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ 이전", disabled=(st.session_state.detail_page == 0), use_container_width=True):
            st.session_state.detail_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px'>{st.session_state.detail_page + 1} / {total_pages} 페이지 ({start+1}–{end} / {total}장)</div>",
            unsafe_allow_html=True
        )
    with col_next:
        if st.button("다음 ▶", disabled=(st.session_state.detail_page >= total_pages - 1), use_container_width=True):
            st.session_state.detail_page += 1
            st.rerun()