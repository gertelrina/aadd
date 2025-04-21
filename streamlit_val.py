import os
import time
import io
import glob
import tempfile
from collections import defaultdict

import librosa
import librosa.display
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import plotly.express as px  # for interactive charts


# === Model and Utilities ===
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 8)
    model_path = "/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-03-11_12-01-06.pth"
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


CLASSES = [
    "AddGaussianNoise", "PitchShift", "TimeStretch", "Shift",
    "ClippingDistortion", "LowPassFilter", "HighPassFilter", "None"
]

# Suppose examples are located in the "examples" folder
EXAMPLES_FOLDER = "//home/user/agertel/dipl/data/tmp_flac/sample_aug/png"

# Defining a list of our images (title + filename in the folder).
# The first is clean (original), the rest are various degradations.
degradation_examples = [
    ("Gauss.Noise", "AddGaussianNoise.png"),
    ("PitchShift", "PitchShift.png"),
    ("TimeStretch", "TimeStretch.png"),
    ("Clipp.Distor", "ClippingDistortion.png"),
    ("LowPass", "LowPassFilter.png"),
    ("HighPass", "HighPassFilter.png"),
    ("Clean", "orig.png"),
]


@st.cache_resource
def get_model():
    return load_model()


def extract_melspectrogram(audio, sr, n_mels=128):
    """Create and save a mel-spectrogram to a temporary file, and return its path."""
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 5))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr)
    plt.axis("off")

    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buf.name, bbox_inches="tight", pad_inches=0)
    plt.close()
    return buf.name


def process_audio(file_path, threshold=0.5, mode="simple"):
    """
    Analyzes a single file: splits it into one-second segments,
    extracts mel-spectrograms, runs inference, and collects results.
    
    Returns (results, x, probs) where:
      - results: a dictionary for display
      - x: a batch of tensors (for possible display of mel-spectrograms)
      - probs: raw probabilities (after sigmoid) of shape [num_segments, num_classes].
    """
    y, sr = librosa.load(file_path, sr=None)
    model = get_model()

    segment_duration = 1  # segment duration (in seconds)
    samples_per_segment = sr * segment_duration
    n_segments = len(y) // samples_per_segment
    if n_segments == 0:
        n_segments = 1

    transform = transforms.ToTensor()
    segments = []

    for i in range(n_segments):
        segment = y[i * samples_per_segment: (i + 1) * samples_per_segment]
        mel_path = extract_melspectrogram(segment, sr)
        mel_img = Image.open(mel_path).convert("RGB")
        os.remove(mel_path)
        segments.append(transform(mel_img))

    x = torch.stack(segments)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.sigmoid(outputs).cpu().numpy()

    result = defaultdict(list)

    if mode == "simple":
        # Maximum across all segments
        avg_probs = np.max(probs, axis=0)
        for idx, p in enumerate(avg_probs):
            if p >= threshold and CLASSES[idx] != "None":
                result[CLASSES[idx]].append(round(float(p), 3))
    else:
        # Detailed (line by line for each segment)
        for i, frame in enumerate(probs):
            second_result = []
            for idx, p in enumerate(frame):
                if p >= threshold and CLASSES[idx] != "None":
                    second_result.append((CLASSES[idx], round(float(p), 3)))
            if second_result:
                result[f"Second {i}"].extend(second_result)

    return result, x, probs


def tensor_to_pil(tensor):
    """Convert a PyTorch Tensor to a PIL image."""
    tensor = tensor.cpu()
    transform_img = T.ToPILImage()
    return transform_img(tensor)


# === Streamlit Configuration ===
st.set_page_config(
    page_title="Audio Degradation Detection",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Audio Degradation Detection")

# Split the screen into two columns: left (1) and right (2).
left_col, right_col = st.columns([1, 2])

# ===== LEFT COLUMN: Examples and inference queue =====
with left_col:
    st.markdown("## Examples for Analysis")
    example_files = [
        "/home/user/agertel/dipl/data/asr_flac_public_phone/8/27/0a6513a19ecc.flac",
        "/home/user/agertel/dipl/data/tmp_flac/2ab251ffbd41.flac"
    ]

    selected_example = st.selectbox("Choose an example for analysis", example_files)

    if "selected_examples" not in st.session_state:
        st.session_state["selected_examples"] = []

    if selected_example:
        st.audio(selected_example, format="audio/flac")
        st.caption(os.path.basename(selected_example))

        if st.button("➕ Add to inference queue", key="add_example"):
            if selected_example not in st.session_state["selected_examples"]:
                st.session_state["selected_examples"].append(selected_example)
                placeholder = st.empty()
                with placeholder:
                    st.success(f"File {os.path.basename(selected_example)} has been added to the inference queue")
                    time.sleep(1)
                placeholder.empty()

    st.markdown("---")
    st.markdown("## Current Inference Queue")

    # List of files in the queue
    with st.expander("📫 File list"):
        if st.session_state["selected_examples"]:
            if st.button("Select All", key="select_all"):
                for file in st.session_state["selected_examples"]:
                    st.session_state[f"checkbox_{file}"] = True

            selected_for_inference = []
            for file in st.session_state["selected_examples"]:
                # If checkbox is checked, add file to analysis list
                if st.checkbox(os.path.basename(file), key=f"checkbox_{file}"):
                    selected_for_inference.append(file)
        else:
            st.write("📭 The queue is empty.")

    with st.expander("✨ Examples of audio degradations"):
        # Split the list into blocks of 2 items (for 2 images per row)
        cols_per_row = 2
        for i in range(0, len(degradation_examples), cols_per_row):
            row_data = degradation_examples[i: i + cols_per_row]
            columns = st.columns(len(row_data))

            for (label, fname), col in zip(row_data, columns):
                image_path = os.path.join(EXAMPLES_FOLDER, fname)
                if os.path.exists(image_path):
                    col.subheader(label)
                    col.image(image_path)
                else:
                    col.warning(f"File {fname} not found")

# ===== RIGHT COLUMN: File uploads and inference =====
with right_col:
    import uuid

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = str(uuid.uuid4())

    # Uploading files with file_uploader
    with st.expander("Upload audio files"):
        if st.button("🧹 Clear uploaded files"):
            st.session_state.upload_key = str(uuid.uuid4())  # reset key so file_uploader clears
            st.rerun()

        uploaded_files = st.file_uploader(
            "Upload audio files", type=["flac", "wav"], accept_multiple_files=True,
            key=st.session_state.upload_key
        )

    # Input path to a local folder
    with st.expander("Or enter a path to a local folder with audio files"):
        folder_path = '/home/user/agertel/dipl/data/tmp_flac'  # Example default
        refresh = st.button("🔄 Load from folder")

    if "audio_paths" not in st.session_state:
        st.session_state["audio_paths"] = []

    if refresh and folder_path:
        audio_paths = glob.glob(os.path.join(folder_path, "**", "*.flac"), recursive=True)
        st.session_state["audio_paths"] = audio_paths
        for x in audio_paths:
            if x not in st.session_state["selected_examples"]:
                st.session_state["selected_examples"].append(x)

        placeholder = st.empty()
        with placeholder:
            st.success(f"Found {len(audio_paths)} files in the folder")
            time.sleep(2)
        placeholder.empty()
        st.rerun()

    elif refresh:
        st.session_state["audio_paths"] = []
        placeholder = st.empty()
        with placeholder:
            st.success("Empty (or incorrect) folder!")
            time.sleep(2)
        placeholder.empty()

    st.markdown("---")
    st.markdown("### ⚙️ Inference Settings")

    threshold = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01, key="main_threshold")

    # Choosing analysis mode and "Show MEL" flag
    cols_settings = st.container()
    with cols_settings:
        c_left, c_mid, c_right = st.columns([1, 1, 1])
        with c_left:
            analysis_type = st.radio(
                "Choose analysis mode",
                ["Simple summary", "Detailed per-second analysis"],
                key="main_mode"
            )
        with c_mid:
            mels_show = st.checkbox("Show MEL", value=False, key="checkbox_mels")

    # Button to clear the inference queue
    if st.button("🧹 Clear inference queue"):
        st.session_state["selected_examples"] = []
        for key in list(st.session_state.keys()):
            if key.startswith("checkbox_"):
                del st.session_state[key]
        st.session_state["audio_paths"] = []

        placeholder = st.empty()
        with placeholder:
            st.success("Clearing inference queue")
            time.sleep(1)
        placeholder.empty()

        st.rerun()

    # Collect files checked in the checkboxes
    file_list = []
    selected_for_inference = []

    for file in st.session_state.get("selected_examples", []):
        if st.session_state.get(f"checkbox_{file}", False):
            selected_for_inference.append(file)

    file_list.extend(selected_for_inference)
    if uploaded_files:
        file_list.extend(uploaded_files)

    # --------------------------------------------------------------------
    # Prepare structure for global statistics
    # --------------------------------------------------------------------
    global_stats = {
        "total_segments": 0,        # Total number of segments (across all audio)
        "dirty_segments": 0,        # Segments with at least 1 degradation
        "counts_per_degradation": {c: 0 for c in CLASSES if c != "None"},  # Counter by type of degradation
        "dirty_audio": 0,           # Audio files with at least one degraded segment
        "clean_audio": 0,           # Audio files with no detected degradation in any segment
        "total_audios": 0           # Total audio files in inference
    }
    # --------------------------------------------------------------------

    if file_list:
        with st.expander("Detailed inference"):
            for uploaded_file in file_list:
                # If the file is a string (path)
                if isinstance(uploaded_file, str):
                    file_path = uploaded_file
                    st.subheader(f"File: {os.path.basename(file_path)}")
                    st.audio(file_path, format="audio/flac")

                # If the file was uploaded via file_uploader
                else:
                    st.subheader(f"File: {uploaded_file.name}")
                    audio_bytes = uploaded_file.read()
                    st.audio(audio_bytes, format="audio/flac")

                    # Save to a temporary location for librosa
                    file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                mode = "simple" if analysis_type == "Simple summary" else "detailed"
                results, mels_tensor, probs = process_audio(file_path, threshold=threshold, mode=mode)

                # ==============================
                # 1) Collect global statistics
                # ==============================
                global_stats["total_audios"] += 1
                n_segments = probs.shape[0]
                global_stats["total_segments"] += n_segments

                audio_has_degrade = False
                # Iterate over all segments and classes
                for row in probs:
                    segment_has_degrade = False
                    for i, p in enumerate(row):
                        # Skip the "None" class
                        if CLASSES[i] == "None":
                            continue
                        # If probability is above threshold
                        if p >= threshold:
                            segment_has_degrade = True
                            global_stats["counts_per_degradation"][CLASSES[i]] += 1
                    if segment_has_degrade:
                        global_stats["dirty_segments"] += 1
                        audio_has_degrade = True

                if audio_has_degrade:
                    global_stats["dirty_audio"] += 1
                else:
                    global_stats["clean_audio"] += 1

                # ==============================
                # 2) Display MEL (if needed)
                # ==============================
                if mels_show:
                    st.markdown("#### Displaying Mel-Spectrograms")
                    max_to_show = 5
                    cols_mels = st.columns(max_to_show)
                    for i in range(min(mels_tensor.shape[0], max_to_show)):
                        img = tensor_to_pil(mels_tensor[i])
                        with cols_mels[i]:
                            st.image(img, caption=f"Sec {i}", use_container_width=True)

                # ==============================
                # 3) Output results for the file
                # ==============================
                st.write("**Detected distortions:**")
                if results:
                    for k, v in results.items():
                        # If v is a list of tuples (detailed mode)
                        if isinstance(v, list) and v and isinstance(v[0], tuple):
                            st.markdown(f"<b>{k}</b>", unsafe_allow_html=True)
                            for cls_name, prob in v:
                                st.markdown(
                                    f"<span style='margin-left: 1em;'>• <b>{cls_name}</b>: {prob:.3f}</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            # For simple mode: "class: probability"
                            st.markdown(f"• <b>{k}</b>: {v[0]:.3f}", unsafe_allow_html=True)
                else:
                    st.write("No detected distortions above the threshold.")

        # ----------------------------------------------------------------
        # Block with overall metrics and interactive charts
        # ----------------------------------------------------------------
        st.markdown("---")
        st.header("Global statistics for all audio")

        total_segments = global_stats["total_segments"]
        dirty_segments = global_stats["dirty_segments"]
        total_audios = global_stats["total_audios"]
        dirty_audios = global_stats["dirty_audio"]
        clean_audios = global_stats["clean_audio"]

        ratio_dirty_segments = dirty_segments / total_segments if total_segments > 0 else 0
        ratio_dirty_audios = dirty_audios / total_audios if total_audios > 0 else 0

        st.write(f"**Total audio files:** {total_audios}")
        st.write(f"**Total segments:** {total_segments}")
        st.write(
            f"**Degraded segments:** {dirty_segments} "
            f"({ratio_dirty_segments:.2%} of the total number of segments)"
        )
        st.write(
            f"**Degraded audio files:** {dirty_audios} "
            f"({ratio_dirty_audios:.2%} of the total number of audio files)"
        )
        st.write(f"**Clean audio files:** {clean_audios}")

        # Build data for charts
        total_found_degradations = sum(global_stats["counts_per_degradation"].values())
        degrade_names = []
        degrade_counts = []
        for degrade, count in global_stats["counts_per_degradation"].items():
            degrade_names.append(degrade)
            degrade_counts.append(count)

        # ---- Bar chart: Distribution of detected degradations ----
        if total_found_degradations > 0:
            fig_bar = px.bar(
                x=degrade_names,
                y=degrade_counts,
                title="Distribution of detected degradations (total count across segments)",
                labels={"x": "Degradation type", "y": "Frequency"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("No degradations were detected, skipping distribution chart.")

        # ---- Pie chart: Degraded vs. Clean Audio ----
        fig_pie_audios = px.pie(
            names=["Degraded audio", "Clean audio"],
            values=[dirty_audios, clean_audios],
            title="Ratio of Degraded vs. Clean Audio"
        )
        st.plotly_chart(fig_pie_audios, use_container_width=True)

        # ---- Pie chart: Degraded vs. Clean Segments ----
        clean_segments = total_segments - dirty_segments
        fig_pie_segments = px.pie(
            names=["Degraded segments", "Clean segments"],
            values=[dirty_segments, clean_segments],
            title="Ratio of Degraded vs. Clean Segments"
        )
        st.plotly_chart(fig_pie_segments, use_container_width=True)

    else:
        st.write("No audio files selected.")
