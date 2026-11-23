import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyedflib
import tempfile
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. CONFIG (paste your NEW keys here)
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "tusz-clinical-eeg-index"
MODEL_WEIGHTS_PATH = "tusz_crossmodal_model.pt"

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ==========================================
# 2. MODEL
# ==========================================
class EEGInferenceTower(nn.Module):
    def __init__(self, eeg_channels=19, embed_dim=256):
        super().__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.AdaptiveAvgPool1d(1)
        )
        self.eeg_projection = nn.Sequential(
            nn.Linear(256, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, eeg_signals):
        feat = self.eeg_encoder(eeg_signals).squeeze(-1)
        return F.normalize(self.eeg_projection(feat), p=2, dim=1)

@st.cache_resource
def load_model():
    model = EEGInferenceTower()
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ==========================================
# 3. EXTRACTION (matches the Colab indexing code)
# ==========================================
def process_edf_file(edf_path, onset_sec=0.0, num_channels=19, max_eeg_len=256):
    f = pyedflib.EdfReader(edf_path)
    n_ch = f.signals_in_file
    sfreq = int(f.getSampleFrequency(0)) if n_ch > 0 else 250
    n_samples = f.getNSamples()
    start_sample = max(0, int(onset_sec * sfreq))

    channel_data = []
    for ch_idx in range(min(n_ch, num_channels)):
        read_start = start_sample if (start_sample + max_eeg_len) < n_samples[ch_idx] else 0
        sig = f.readSignal(ch_idx, start=read_start, n=max_eeg_len)
        if len(sig) < max_eeg_len:
            sig = np.pad(sig, (0, max_eeg_len - len(sig)), mode="constant")
        channel_data.append(sig[:max_eeg_len])
    f.close()

    while len(channel_data) < num_channels:
        channel_data.append(np.zeros(max_eeg_len, dtype=np.float32))

    eeg = torch.tensor(np.array(channel_data), dtype=torch.float32)
    mean = eeg.mean(dim=1, keepdim=True)
    std = eeg.std(dim=1, keepdim=True) + 1e-6
    eeg = (eeg - mean) / std
    return eeg.unsqueeze(0)          # returns ONE value

# ==========================================
# 4. UI
# ==========================================
from collections import defaultdict

DUMMY_VEC = [1.0] * 256   # only used to look up a record by filename

def lookup_ground_truth(filename):
    """Find the uploaded file's own record in the index to get its true label and onset."""
    res = index.query(
        vector=DUMMY_VEC, top_k=1, include_metadata=True,
        filter={"edf_file": {"$eq": filename}},
    )
    return res["matches"][0].get("metadata", {}) if res["matches"] else None

st.title("Cross-Modal Clinical EEG Retrieval System")
st.write("Upload a raw .edf file to query Pinecone and generate an evidence-based clinical report.")

uploaded_file = st.file_uploader("Upload TUSZ EEG File (.edf)", type=["edf"])

if uploaded_file is not None:
    # Patient ID comes from the filename, e.g. aaaaatth_s001_t000.edf -> aaaaatth
    query_patient_id = uploaded_file.name.split("_")[0]

    truth = lookup_ground_truth(uploaded_file.name)
    onset = float(truth.get("onset_sec", 0.0)) if truth else 0.0

    with st.spinner("Extracting waveform and computing 256-d vector..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        try:
            eeg_tensor = process_edf_file(temp_path, onset_sec=onset)
        finally:
            os.remove(temp_path)

        with torch.no_grad():
            embedding = model(eeg_tensor).squeeze().numpy().tolist()

    with st.spinner("Querying Pinecone Vector Database..."):
        query_response = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
            filter={"patient_id": {"$ne": query_patient_id}},   # silent exclusion
        )
        matches = query_response["matches"]

    st.subheader("Top 5 Historical Matches Retrieved")
    context_string = ""
    votes = defaultdict(float)

    for idx, match in enumerate(matches):
        sim_score = match["score"]
        meta = match.get("metadata", {})
        label = meta.get("diagnosis", "Unknown")
        code = meta.get("seizure_code", "")
        patient_id = meta.get("patient_id", "Unknown")
        note = meta.get("clinical_note", "No note available.")

        votes[label] += sim_score
        st.write(f"**{idx+1}. {label} ({code})** (Similarity: {sim_score:.4f}) - Patient ID: {patient_id}")
        context_string += f"Match {idx+1}: Diagnosis: {label} ({code}), Score: {sim_score:.4f}, Note: {note}\n"

    # ---------- Actual vs. System Result ----------
    st.subheader("Actual Activity vs. System Result")
    predicted = max(votes, key=votes.get) if votes else "Unknown"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Actual activity of this recording**")
        if truth:
            st.write(f"{truth.get('diagnosis', 'Unknown')} ({truth.get('seizure_code', '')})")
            st.caption(f"Annotated onset: {truth.get('onset_sec', 0):.1f} s")
        else:
            st.write("Not available for this file")
    with col2:
        st.markdown("**System result (consensus of top 5)**")
        st.write(predicted)
        st.caption(f"Weighted vote across {len(matches)} retrieved cases")

    if truth:
        if truth.get("diagnosis") == predicted:
            st.success("The retrieved cases agree with the actual activity.")
        else:
            st.warning("The retrieved cases do not match the actual activity.")

    st.subheader("AI-Generated Clinical Report")
    with st.spinner("Synthesizing report via Gemini..."):
        prompt = f"""You are an expert clinical epileptologist. Generate a 3-part report based strictly on the retrieved similar cases below. Do not invent measurements that are not in the data.
Structure it exactly like this:
**1. Findings:** Summarize the signal characteristics described in the retrieved cases.
**2. Historical Evidence:** Summarize the top retrieved matches and their similarity scores.
**3. Clinical Impression:** Give a suggested diagnosis based on the consensus of the matches, and note any disagreement among them.

DATA CONTEXT:
{context_string}"""
        response = llm.generate_content(prompt)
        st.success("Report Generated Successfully!")
        st.write(response.text)
        st.caption("Research prototype. Not a medical device and not for clinical decisions.")