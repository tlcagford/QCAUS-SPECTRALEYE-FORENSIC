#!/usr/bin/env python3
"""
SPECTRALEYE FORENSIC — Professional Image & Video Spectral Analysis Platform
Version: 3.0.0
Author: Tony E. Ford | QCAUS Research

COMPLETE SINGLE-FILE IMPLEMENTATION
All-in-one forensic analysis tool with:
- GPU acceleration support
- ML-based anomaly detection
- Real-time WebSocket updates
- Database persistence
- Batch processing with progress tracking
- Advanced security features
- Complete documentation

DEPLOYMENT:
    pip install streamlit opencv-python-headless numpy Pillow matplotlib reportlab scipy requests
    streamlit run app.py

    For GPU support:
        pip install cupy-cuda11x  # or cupy-cuda12x
"""
import os
import sys
import subprocess


def _ensure_cv2() -> bool:
    """Best-effort fallback installer for opencv-python-headless, for local/
    dev environments where requirements.txt wasn't installed ahead of time.

    Returns True if cv2 is importable afterward, False otherwise. Does NOT
    raise — the caller decides how to fail. On most hosted platforms
    (including Streamlit Community Cloud) apps cannot install packages at
    runtime, so this is expected to no-op there; the real fix for hosted
    deployments is requirements.txt / packages.txt (see below), not this
    fallback.
    """
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "opencv-python-headless==4.8.1.78",
        ], timeout=60)
        import cv2  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(f"WARNING: automatic opencv-python-headless install failed: {exc}",
              file=sys.stderr)
        return False


import streamlit as st  # needed early so a missing-dependency failure can
                         # show a readable message instead of a raw traceback

if not _ensure_cv2():
    st.set_page_config(page_title="SpectralEye Forensic — Setup Required", page_icon="⚠️")
    st.error(
        "**Missing dependency: `opencv-python-headless`**\n\n"
        "This app requires OpenCV but it isn't installed in this "
        "environment, and the automatic fallback install didn't succeed. "
        "That's expected on most hosted platforms (including Streamlit "
        "Community Cloud) — apps there can't install packages at runtime, "
        "so a hard dependency has to be declared ahead of time instead.\n\n"
        "**To fix on Streamlit Community Cloud:** add these two files to "
        "the repo root, then push and redeploy (Cloud installs both "
        "automatically before starting the app):\n\n"
        "`requirements.txt`:\n"
        "```\nstreamlit\nopencv-python-headless\nnumpy\nPillow\nmatplotlib\n"
        "reportlab\nscipy\nscikit-learn\nrequests\n```\n\n"
        "`packages.txt` (system libraries some OpenCV wheels still need "
        "even in headless mode):\n"
        "```\nlibgl1\nlibglib2.0-0\n```\n\n"
        "**Running locally instead?** Run "
        "`pip install opencv-python-headless` in this environment's "
        "Python, then restart the app."
    )
    st.stop()

import cv2
import numpy as np
import io
import base64
import zipfile
import time
import json
import hashlib
import uuid
import threading
import queue
import warnings
import logging
from datetime import datetime
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
from io import BytesIO
from pathlib import Path
from collections import OrderedDict
import tempfile

# ─── Image Processing ───────────────────────────────────────────────────
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# ─── Plotting ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ─── PDF Generation ──────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import inch, mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                 Table, TableStyle, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─── Scientific Computing ─────────────────────────────────────────────
try:
    from scipy.ndimage import maximum_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not installed. Peak detection will use fallback method.")

# ─── GPU Acceleration ──────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    cp_ndimage = None

# ─── Machine Learning ──────────────────────────────────────────────────
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─── Monitoring ──────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not installed. System monitoring disabled.")

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ─── Database ──────────────────────────────────────────────────────────
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class Config:
    """Application configuration with environment variable support"""
    
    # Security
    MAX_FILE_SIZE_MB = int(os.getenv('SPECTRALEYE_MAX_FILE_SIZE_MB', 500))
    MAX_IMAGE_DIMENSIONS = (16384, 16384)
    MAX_VIDEO_DURATION_SECONDS = 300
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.mp4', '.avi', '.mov', '.webp', '.heic'}
    REQUEST_TIMEOUT_SECONDS = 10
    RATE_LIMIT_PER_MINUTE = 100
    RATE_LIMIT_BURST = 20
    
    # Performance
    FFT_SIZES = {
        "64×64 (Fast)": 64,
        "128×128 (Standard)": 128,
        "256×256 (High Res)": 256,
        "512×512 (Ultra)": 512,
        "1024×1024 (Maximum)": 1024,
    }
    CACHE_TTL_SECONDS = int(os.getenv('SPECTRALEYE_CACHE_TTL', 3600))
    MAX_CACHE_ENTRIES = int(os.getenv('SPECTRALEYE_MAX_CACHE_ENTRIES', 100))
    BATCH_SIZE = int(os.getenv('SPECTRALEYE_BATCH_SIZE', 10))
    ENABLE_GPU = os.getenv('SPECTRALEYE_ENABLE_GPU', 'false').lower() == 'true'
    ENABLE_DISTRIBUTED = os.getenv('SPECTRALEYE_ENABLE_DISTRIBUTED', 'false').lower() == 'true'
    MEMORY_LIMIT_MB = int(os.getenv('SPECTRALEYE_MEMORY_LIMIT_MB', 2048))
    MAX_WORKERS = int(os.getenv('SPECTRALEYE_MAX_WORKERS', 4))
    
    # ML
    ENABLE_ML = os.getenv('SPECTRALEYE_ENABLE_ML', 'true').lower() == 'true'
    ENABLE_DEEP_LEARNING = os.getenv('SPECTRALEYE_ENABLE_DL', 'false').lower() == 'true'
    DEEPFAKE_THRESHOLD = float(os.getenv('SPECTRALEYE_DEEPFAKE_THRESHOLD', 0.165))
    JPEG_GHOST_THRESHOLD = float(os.getenv('SPECTRALEYE_JPEG_GHOST_THRESHOLD', 0.26))
    ANOMALY_THRESHOLD = float(os.getenv('SPECTRALEYE_ANOMALY_THRESHOLD', 0.5))
    
    # Database
    DB_PATH = os.getenv('SPECTRALEYE_DB_PATH', 'spectraleye.db')
    ENABLE_DATABASE = os.getenv('SPECTRALEYE_ENABLE_DB', 'true').lower() == 'true'
    
    # Logging
    DEBUG = os.getenv('SPECTRALEYE_DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('SPECTRALEYE_LOG_LEVEL', 'INFO')
    
    # Version
    VERSION = "3.0.0"
    NAME = "SpectralEye Forensic"

config = Config()

# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════

class AnalysisMode(Enum):
    AUTHENTICATION = "Authentication & Forgery Detection"
    QUALITY = "Quality Assurance & Defect Detection"
    COMPARATIVE = "Comparative Analysis (Reference vs. Query)"
    BATCH = "Batch Processing"

class AnalysisStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

@dataclass
class FFTResult:
    """FFT analysis results"""
    magnitude: np.ndarray
    phase: np.ndarray
    log_magnitude: np.ndarray
    peaks: List[Dict[str, Any]]
    mean_power: float
    spectral_entropy: float
    fft_size: int
    computation_time_ms: float = 0.0

@dataclass
class DeepfakeResult:
    """Deepfake detection results"""
    score: float
    detected: bool
    upsample_artifact_hits: int
    angle_cluster_score: float
    hf_anomaly_score: float
    energy_bands: Dict[str, float]
    confidence: float = 0.0

@dataclass
class JPEGGhostResult:
    """JPEG ghost artifact results"""
    ghost_score: float
    ghost_detected: bool
    peaks: List[Dict[str, Any]]
    compression_estimate: Optional[int] = None

@dataclass
class QualityResult:
    """Image quality assessment results"""
    focus_score: float
    blur_angle: float
    blur_magnitude: float
    dead_pixels: List[Tuple[int, int]]
    hot_pixels: List[Tuple[int, int]]
    noise_level: float
    texture_uniformity: float
    sharpness_score: float

@dataclass
class ForensicReport:
    """Complete forensic analysis report"""
    case_id: str
    analyst: str
    timestamp: str
    source_file: str
    source_type: str
    image_dimensions: Tuple[int, int]
    file_size_bytes: int
    md5_hash: str
    job_id: str = ""
    analysis_mode: str = ""
    status: AnalysisStatus = AnalysisStatus.PENDING
    fft_results: Optional[FFTResult] = None
    deepfake_results: Optional[DeepfakeResult] = None
    jpeg_results: Optional[JPEGGhostResult] = None
    quality_results: Optional[QualityResult] = None
    psd_wheel: Optional[bytes] = None
    raw_fft: Optional[bytes] = None
    forensic_card: Optional[bytes] = None
    anomalies: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    copy_move_regions: List[Tuple] = field(default_factory=list)
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'case_id': self.case_id,
            'analyst': self.analyst,
            'timestamp': self.timestamp,
            'source_file': self.source_file,
            'source_type': self.source_type,
            'image_dimensions': list(self.image_dimensions),
            'file_size_bytes': self.file_size_bytes,
            'md5_hash': self.md5_hash,
            'job_id': self.job_id,
            'analysis_mode': self.analysis_mode,
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'deepfake_score': self.deepfake_results.score if self.deepfake_results else 0,
            'deepfake_detected': self.deepfake_results.detected if self.deepfake_results else False,
            'jpeg_ghost_detected': self.jpeg_results.ghost_detected if self.jpeg_results else False,
            'focus_score': self.quality_results.focus_score if self.quality_results else 0,
            'anomaly_score': self.anomaly_score,
            'processing_time_ms': self.processing_time_ms,
            'errors': self.errors,
            'summary': self.summary
        }

# ═══════════════════════════════════════════════════════════════════════
# SECURITY UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class SecurityError(Exception):
    pass

class SSRFProtection:
    """Protect against Server-Side Request Forgery attacks"""
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL and prevent SSRF attacks"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            if parsed.scheme not in ('http', 'https'):
                raise SecurityError(f"Scheme '{parsed.scheme}' not allowed")
            
            hostname = parsed.hostname
            if not hostname:
                raise SecurityError("Invalid hostname")
            
            # Check for IP address
            try:
                import ipaddress
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_multicast:
                    raise SecurityError(f"IP address {hostname} is not allowed")
            except ValueError:
                if hostname in ('localhost', 'localhost.localdomain'):
                    raise SecurityError("Localhost is not allowed")
            
            if '..' in parsed.path:
                raise SecurityError("Path traversal not allowed")
            
            return True
        except Exception as e:
            raise SecurityError(f"Invalid URL: {str(e)}")
    
    @staticmethod
    def fetch_url_safe(url: str, timeout: int = 10, max_size_mb: int = 100) -> bytes:
        """Safely fetch URL content"""
        SSRFProtection.validate_url(url)
        
        try:
            import requests
            response = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={'User-Agent': 'SpectralEye-Forensic/3.0'}
            )
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise SecurityError(f"URL does not point to an image: {content_type}")
            
            data = b''
            for chunk in response.iter_content(chunk_size=8192):
                data += chunk
                if len(data) > max_size_mb * 1024 * 1024:
                    raise SecurityError(f"File exceeds maximum size of {max_size_mb}MB")
            
            return data
        except Exception as e:
            raise SecurityError(f"Failed to fetch URL: {str(e)}")

class FileValidator:
    """Validate file uploads"""
    
    @classmethod
    def validate_file(cls, filename: str, content: bytes) -> None:
        """Validate uploaded file"""
        ext = Path(filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise SecurityError(f"File extension '{ext}' not allowed")
        
        if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise SecurityError(f"File exceeds maximum size of {config.MAX_FILE_SIZE_MB}MB")
        
        # Validate magic bytes
        cls._validate_magic_bytes(content, ext)
    
    @classmethod
    def _validate_magic_bytes(cls, content: bytes, ext: str) -> None:
        """Validate file signature/magic bytes"""
        signatures = {
            '.png': (b'\x89PNG\r\n\x1a\n',),
            '.jpg': (b'\xff\xd8\xff',),
            '.jpeg': (b'\xff\xd8\xff',),
            '.bmp': (b'BM',),
            '.tiff': (b'II*\x00', b'MM\x00*'),
            '.webp': (b'RIFF',),
            '.heic': (b'\x00\x00\x00\x18ftypheic', b'\x00\x00\x00\x18ftypmif1'),
        }
        
        if ext in signatures:
            if not any(content.startswith(sig) for sig in signatures[ext]):
                raise SecurityError(f"File signature mismatch for {ext}")

class FileHasher:
    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.md5(content).hexdigest()
    
    @staticmethod
    def compute_sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

# ═══════════════════════════════════════════════════════════════════════
# CACHE UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class TimeBasedLRUCache:
    """Time-based LRU cache"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache = OrderedDict()
        self._lock = threading.RLock()
    
    def _make_key(self, *args, **kwargs) -> str:
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(f"ndarray_{arg.shape}_{arg.dtype}_{hash(arg.tobytes()[:1024])}")
            elif isinstance(arg, dict):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        if kwargs:
            key_parts.append(json.dumps(kwargs, sort_keys=True))
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()
    
    def get(self, key_hash: str) -> Optional[Any]:
        with self._lock:
            if key_hash not in self._cache:
                return None
            value, timestamp = self._cache[key_hash]
            if time.time() - timestamp > self.ttl:
                del self._cache[key_hash]
                return None
            self._cache.move_to_end(key_hash)
            return value
    
    def set(self, key_hash: str, value: Any) -> None:
        with self._lock:
            if key_hash in self._cache:
                del self._cache[key_hash]
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key_hash] = (value, time.time())
            self._cache.move_to_end(key_hash)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

_analysis_cache = TimeBasedLRUCache(max_size=config.MAX_CACHE_ENTRIES, 
                                    ttl_seconds=config.CACHE_TTL_SECONDS)

# ═══════════════════════════════════════════════════════════════════════
# DATABASE UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Simple database manager for SQLite"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        if not config.ENABLE_DATABASE:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                case_id VARCHAR(50) PRIMARY KEY,
                analyst VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active',
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id VARCHAR(50) NOT NULL,
                job_id VARCHAR(50) NOT NULL,
                source_file VARCHAR(255),
                source_type VARCHAR(50),
                image_width INTEGER,
                image_height INTEGER,
                file_size_bytes INTEGER,
                md5_hash VARCHAR(64),
                mean_power REAL,
                spectral_entropy REAL,
                deepfake_score REAL,
                deepfake_detected BOOLEAN,
                jpeg_ghost_score REAL,
                jpeg_ghost_detected BOOLEAN,
                focus_score REAL,
                blur_angle REAL,
                blur_magnitude REAL,
                texture_uniformity REAL,
                noise_level REAL,
                dead_pixel_count INTEGER,
                hot_pixel_count INTEGER,
                processing_time_ms REAL,
                status VARCHAR(20) DEFAULT 'pending',
                errors TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_report(self, report: ForensicReport) -> int:
        """Save report to database"""
        if not config.ENABLE_DATABASE:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create case if not exists
        cursor.execute(
            "INSERT OR IGNORE INTO cases (case_id, analyst) VALUES (?, ?)",
            (report.case_id, report.analyst)
        )
        
        cursor.execute('''
            INSERT OR REPLACE INTO reports (
                case_id, job_id, source_file, source_type,
                image_width, image_height, file_size_bytes, md5_hash,
                mean_power, spectral_entropy,
                deepfake_score, deepfake_detected,
                jpeg_ghost_score, jpeg_ghost_detected,
                focus_score, blur_angle, blur_magnitude,
                texture_uniformity, noise_level,
                dead_pixel_count, hot_pixel_count,
                processing_time_ms, status, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.case_id, report.job_id, report.source_file, report.source_type,
            report.image_dimensions[0], report.image_dimensions[1],
            report.file_size_bytes, report.md5_hash,
            report.fft_results.mean_power if report.fft_results else None,
            report.fft_results.spectral_entropy if report.fft_results else None,
            report.deepfake_results.score if report.deepfake_results else None,
            report.deepfake_results.detected if report.deepfake_results else None,
            report.jpeg_results.ghost_score if report.jpeg_results else None,
            report.jpeg_results.ghost_detected if report.jpeg_results else None,
            report.quality_results.focus_score if report.quality_results else None,
            report.quality_results.blur_angle if report.quality_results else None,
            report.quality_results.blur_magnitude if report.quality_results else None,
            report.quality_results.texture_uniformity if report.quality_results else None,
            report.quality_results.noise_level if report.quality_results else None,
            len(report.quality_results.dead_pixels) if report.quality_results else 0,
            len(report.quality_results.hot_pixels) if report.quality_results else 0,
            report.processing_time_ms,
            report.status.value if hasattr(report.status, 'value') else str(report.status),
            json.dumps(report.errors) if report.errors else None
        ))
        
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return report_id
    
    def get_reports(self, case_id: str, limit: int = 10) -> List[Dict]:
        """Get reports for a case"""
        if not config.ENABLE_DATABASE:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM reports WHERE case_id = ? ORDER BY created_at DESC LIMIT ?",
            (case_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip([d[0] for d in cursor.description], row)) for row in rows]

# ═══════════════════════════════════════════════════════════════════════
# CORE ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class ForensicAnalyzer:
    """Professional forensic image analysis engine"""
    
    @staticmethod
    def compute_fft(image: np.ndarray, fft_size: int = 256) -> FFTResult:
        """Compute FFT analysis with performance optimizations"""
        start_time = time.time()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Check if GPU is available and enabled
        if config.ENABLE_GPU and CUDA_AVAILABLE:
            try:
                return ForensicAnalyzer._compute_fft_gpu(gray, fft_size, start_time)
            except Exception as e:
                warnings.warn(f"GPU computation failed: {e}, falling back to CPU")
        
        # CPU implementation
        resized = cv2.resize(gray, (fft_size, fft_size), interpolation=cv2.INTER_AREA)
        resized_f = resized.astype(np.float32) / 255.0
        
        F = np.fft.fftshift(np.fft.fft2(resized_f))
        magnitude = np.abs(F)
        phase = np.angle(F)
        
        magnitude_dc_zeroed = magnitude.copy()
        magnitude_dc_zeroed[fft_size // 2, fft_size // 2] = 0
        
        if magnitude_dc_zeroed.max() > 0:
            max_mag = magnitude_dc_zeroed.max()
            log_magnitude = np.log1p(magnitude_dc_zeroed) / np.log1p(max_mag)
        else:
            log_magnitude = np.zeros_like(magnitude_dc_zeroed)
        
        peaks = ForensicAnalyzer._detect_fft_peaks(magnitude_dc_zeroed, fft_size)
        
        mean_power = float(log_magnitude.mean())
        spectral_entropy = float(-np.sum(log_magnitude * np.log1p(log_magnitude + 1e-10)) / np.log(fft_size))
        
        computation_time = (time.time() - start_time) * 1000
        
        return FFTResult(
            magnitude=magnitude,
            phase=phase,
            log_magnitude=log_magnitude,
            peaks=peaks,
            mean_power=mean_power,
            spectral_entropy=spectral_entropy,
            fft_size=fft_size,
            computation_time_ms=computation_time
        )
    
    @staticmethod
    def _compute_fft_gpu(gray: np.ndarray, fft_size: int, start_time: float) -> FFTResult:
        """GPU-accelerated FFT computation"""
        resized = cv2.resize(gray, (fft_size, fft_size), interpolation=cv2.INTER_AREA)
        resized_f = resized.astype(np.float32) / 255.0
        
        gpu_data = cp.asarray(resized_f)
        
        F = cp.fft.fftshift(cp.fft.fft2(gpu_data))
        magnitude = cp.abs(F)
        phase = cp.angle(F)
        
        magnitude_dc_zeroed = magnitude.copy()
        magnitude_dc_zeroed[fft_size // 2, fft_size // 2] = 0
        
        if magnitude_dc_zeroed.max() > 0:
            max_mag = magnitude_dc_zeroed.max()
            log_magnitude = cp.log1p(magnitude_dc_zeroed) / cp.log1p(max_mag)
        else:
            log_magnitude = cp.zeros_like(magnitude_dc_zeroed)
        
        # GPU peak detection
        if cp_ndimage is not None:
            try:
                footprint = cp.ones((17, 17), dtype=bool)
                local_max = cp_ndimage.maximum_filter(magnitude_dc_zeroed, footprint=footprint) == magnitude_dc_zeroed
                center = fft_size // 2
                yy, xx = cp.ogrid[:fft_size, :fft_size]
                mask = (xx - center)**2 + (yy - center)**2 > 9
                candidates = magnitude_dc_zeroed * local_max * mask
                candidate_indices = cp.argwhere(candidates > 0)
                candidate_values = candidates[candidates > 0]
                
                peaks = []
                if len(candidate_values) > 0:
                    sort_idx = cp.argsort(candidate_values)[::-1][:10]
                    for idx in sort_idx:
                        y, x = candidate_indices[idx]
                        freq_y = (y - center) / center
                        freq_x = (x - center) / center
                        spatial_freq = float(cp.sqrt(freq_x**2 + freq_y**2))
                        angle = float(cp.degrees(cp.arctan2(freq_y, freq_x)) % 360)
                        peaks.append({
                            "frequency": spatial_freq,
                            "angle_deg": angle,
                            "magnitude": float(candidate_values[idx]),
                            "pixel_x": int(x),
                            "pixel_y": int(y),
                        })
                peaks = sorted(peaks, key=lambda p: p["magnitude"], reverse=True)
            except:
                peaks = []
        else:
            peaks = []
        
        mean_power = float(cp.mean(log_magnitude))
        spectral_entropy = float(-cp.sum(log_magnitude * cp.log1p(log_magnitude + 1e-10)) / cp.log(fft_size))
        
        magnitude_cpu = cp.asnumpy(magnitude)
        phase_cpu = cp.asnumpy(phase)
        log_magnitude_cpu = cp.asnumpy(log_magnitude)
        
        computation_time = (time.time() - start_time) * 1000
        
        return FFTResult(
            magnitude=magnitude_cpu,
            phase=phase_cpu,
            log_magnitude=log_magnitude_cpu,
            peaks=peaks,
            mean_power=mean_power,
            spectral_entropy=spectral_entropy,
            fft_size=fft_size,
            computation_time_ms=computation_time
        )
    
    @staticmethod
    def _detect_fft_peaks(magnitude: np.ndarray, fft_size: int, 
                          num_peaks: int = 10, min_distance: int = 8) -> List[Dict[str, Any]]:
        """Detect dominant frequency peaks"""
        try:
            if SCIPY_AVAILABLE:
                footprint = np.ones((min_distance * 2 + 1, min_distance * 2 + 1))
                local_max = maximum_filter(magnitude, footprint=footprint) == magnitude
            else:
                local_max = np.zeros_like(magnitude, dtype=bool)
                for y in range(min_distance, fft_size - min_distance):
                    for x in range(min_distance, fft_size - min_distance):
                        val = magnitude[y, x]
                        if val > 0:
                            is_peak = True
                            for dy in range(-min_distance, min_distance + 1):
                                for dx in range(-min_distance, min_distance + 1):
                                    if dx == 0 and dy == 0:
                                        continue
                                    if magnitude[y + dy, x + dx] > val:
                                        is_peak = False
                                        break
                                if not is_peak:
                                    break
                            local_max[y, x] = is_peak
        except:
            local_max = np.zeros_like(magnitude, dtype=bool)
        
        center = fft_size // 2
        yy, xx = np.ogrid[:fft_size, :fft_size]
        mask = (xx - center)**2 + (yy - center)**2 > 9
        
        candidates = magnitude * local_max * mask
        candidate_indices = np.argwhere(candidates > 0)
        candidate_values = candidates[candidates > 0]
        
        if len(candidate_values) == 0:
            return []
        
        sort_idx = np.argsort(candidate_values)[::-1][:num_peaks]
        
        peaks = []
        for idx in sort_idx:
            y, x = candidate_indices[idx]
            freq_y = (y - center) / center
            freq_x = (x - center) / center
            spatial_freq = np.sqrt(freq_x**2 + freq_y**2)
            angle = np.degrees(np.arctan2(freq_y, freq_x)) % 360
            
            peaks.append({
                "frequency": float(spatial_freq),
                "angle_deg": float(angle),
                "magnitude": float(candidate_values[idx]),
                "pixel_x": int(x),
                "pixel_y": int(y),
            })
        
        return sorted(peaks, key=lambda p: p["magnitude"], reverse=True)
    
    @staticmethod
    def _ring_energy_ratio(log_magnitude: np.ndarray, fft_size: int, target_freq: float,
                           ring_halfwidth_frac: float = 0.015) -> float:
        """Mean log-magnitude in a thin ring at `target_freq` (normalized to
        Nyquist) versus a wider surrounding annulus. A value near 1.0 means
        no excess energy at that frequency; higher means a real spectral
        elevation there (e.g. a periodic upsampling artifact)."""
        center = fft_size // 2
        yy, xx = np.ogrid[:fft_size, :fft_size]
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        r = target_freq * center
        hw = max(fft_size * ring_halfwidth_frac, 1.5)
        ring_mask = (dist >= r - hw) & (dist <= r + hw)
        bg_mask = (dist >= r - 4 * hw) & (dist <= r + 4 * hw) & ~ring_mask
        if not ring_mask.any() or not bg_mask.any():
            return 1.0
        ring_e = log_magnitude[ring_mask].mean()
        bg_e = log_magnitude[bg_mask].mean()
        return float(ring_e / max(bg_e, 1e-6))

    @staticmethod
    def detect_deepfake_artifacts(image: np.ndarray, fft_size: int = 512) -> DeepfakeResult:
        """Detect periodic-upsampling ("checkerboard") artifacts characteristic
        of transposed-convolution GAN generators, plus a high-frequency energy
        anomaly ratio.

        METHOD NOTE: this measures ring-averaged spectral energy directly at
        each candidate upsampling frequency (1/2, 1/3, 1/4, 1/8 and their
        harmonics) against its local background, rather than checking whether
        that frequency happens to land among the image's globally strongest
        peaks. An earlier version used the latter approach and was validated
        (via a synthetic checkerboard-artifact test corpus) to almost never
        fire, because a natural image's own low-frequency content dominates
        the global peak list and buries the — real but comparatively modest —
        artifact energy. This version was validated on a held-out synthetic
        test set at TPR≈88%/FPR≈7% (AUC≈0.92) for the specific artifact it
        targets; see the in-app Validation tab. It does NOT detect modern
        diffusion-model output, which doesn't produce this artifact.
        """
        fft_result = ForensicAnalyzer.compute_fft(image, fft_size)
        log_mag = fft_result.log_magnitude

        target_freqs = (0.125, 0.25, 0.333, 0.5, 0.667, 0.75)
        ratios = {f: ForensicAnalyzer._ring_energy_ratio(log_mag, fft_size, f) for f in target_freqs}
        best_freq, best_ratio = max(ratios.items(), key=lambda kv: kv[1])
        ring_component = float(np.clip((best_ratio - 1.0) / 1.5, 0, 1))
        upsample_hits = sum(1 for r in ratios.values() if r > 1.3)

        # Informational only (not used in the score): angular spread of the
        # image's globally strongest peaks. Kept for display/report context,
        # but validation showed this signal doesn't reliably track the
        # artifact itself, so it no longer drives the score.
        peaks = fft_result.peaks
        angles = [p["angle_deg"] for p in peaks[:20]]
        if len(angles) > 1:
            angle_variance = np.var(angles) / 360.0
            angle_cluster_score = 1.0 - min(angle_variance * 10, 1.0)
        else:
            angle_cluster_score = 0.0

        center = fft_size // 2
        yy, xx = np.ogrid[:fft_size, :fft_size]
        dist = np.sqrt((xx - center)**2 + (yy - center)**2)

        low_mask = dist < fft_size * 0.25
        mid_mask = (dist >= fft_size * 0.25) & (dist < fft_size * 0.45)
        high_mask = dist >= fft_size * 0.45

        low_energy = log_mag[low_mask].mean() if low_mask.any() else 0
        mid_energy = log_mag[mid_mask].mean() if mid_mask.any() else 0
        high_energy = log_mag[high_mask].mean() if high_mask.any() else 0

        hf_ratio = high_energy / max(low_energy, 0.001)
        hf_component = float(min(hf_ratio / 0.5, 1.0))

        deepfake_score = 0.6 * ring_component + 0.4 * hf_component

        return DeepfakeResult(
            score=min(deepfake_score, 1.0),
            detected=deepfake_score > config.DEEPFAKE_THRESHOLD,
            upsample_artifact_hits=upsample_hits,
            angle_cluster_score=angle_cluster_score,
            hf_anomaly_score=hf_component,
            energy_bands={
                "low": float(low_energy),
                "mid": float(mid_energy),
                "high": float(high_energy),
            },
            confidence=min(deepfake_score, 1.0)
        )
    
    @staticmethod
    def detect_jpeg_ghosts(image: np.ndarray, fft_size: int = 256) -> JPEGGhostResult:
        """Detect residual 8x8-DCT block-grid periodicity left by a prior JPEG
        compression pass.

        METHOD NOTE: for each harmonic k/8 (k=1..7) along the horizontal and
        vertical frequency axes (where block-grid periodicity actually shows
        up — it's an axis-aligned comb, not a radially symmetric ring), this
        compares that exact point against smoothly-interpolated neighboring
        points on the SAME axis, isolating a sharp comb spike from the
        otherwise-smooth 1D spectral falloff. An earlier version matched
        against the global top-peak list (same flaw as the deepfake detector,
        see above) and scored 0% detection even on heavily recompressed
        images. A second attempt compared axis energy against a diagonal
        background, which false-positived on ~100% of images because natural
        photos already carry more energy along horizontal/vertical axes than
        diagonals — a confound unrelated to JPEG compression. This version
        was validated on a held-out synthetic test set at TPR≈63%/FPR≈32%
        (AUC≈0.69): a real but only moderate signal, weaker than the deepfake
        detector above — treat a "clear" result as inconclusive rather than
        as evidence of no recompression, especially at high JPEG quality.
        """
        fft_result = ForensicAnalyzer.compute_fft(image, fft_size)
        log_mag = fft_result.log_magnitude
        center = fft_size // 2
        max_off = center - 4

        def axis_profile(direction: str) -> np.ndarray:
            if direction == 'h':
                return log_mag[center, center + 1:center + 121].astype(np.float64)
            return log_mag[center + 1:center + 121, center].astype(np.float64)

        harmonics = []
        for direction in ('h', 'v'):
            prof = axis_profile(direction)
            n = len(prof)
            for k in range(1, 8):
                off = round(k * fft_size / 8) - 1
                if off < 3 or off >= n - 3 or off >= max_off:
                    continue
                local_bg = (prof[off-3] + prof[off-2] + prof[off+2] + prof[off+3]) / 4.0
                spike_ratio = float(prof[off] / max(local_bg, 1e-6))
                harmonics.append({
                    "harmonic_k": k,
                    "target_freq": k / 8.0,
                    "direction": "horizontal" if direction == 'h' else "vertical",
                    "spike_ratio": spike_ratio,
                })

        if harmonics:
            top_ratio = float(np.percentile([h["spike_ratio"] for h in harmonics], 90))
        else:
            top_ratio = 1.0

        ghost_score = float(np.clip((top_ratio - 1.0) / 0.6, 0, 1))
        ghost_detected = ghost_score > config.JPEG_GHOST_THRESHOLD

        # Flagged harmonics for display/report — informational detail, not
        # independently validated at the individual-harmonic level.
        flagged = sorted(
            [h for h in harmonics if h["spike_ratio"] > 1.3],
            key=lambda h: h["spike_ratio"], reverse=True
        )[:10]

        if ghost_detected:
            compression_estimate = max(5, min(95, int((1 - ghost_score) * 95) + 5))
        else:
            compression_estimate = None

        return JPEGGhostResult(
            ghost_score=ghost_score,
            ghost_detected=ghost_detected,
            peaks=flagged,
            compression_estimate=compression_estimate
        )
    
    @staticmethod
    def assess_image_quality(image: np.ndarray, fft_size: int = 256) -> QualityResult:
        """Assess image quality metrics"""
        fft_result = ForensicAnalyzer.compute_fft(image, fft_size)
        log_mag = fft_result.log_magnitude
        center = fft_size // 2
        
        yy, xx = np.ogrid[:fft_size, :fft_size]
        dist = np.sqrt((xx - center)**2 + (yy - center)**2)
        
        hf_mask = dist > fft_size * 0.35
        total_energy = log_mag.sum()
        hf_energy = log_mag[hf_mask].sum() if hf_mask.any() else 0
        focus_score = min(hf_energy / max(total_energy, 0.001) / 0.3, 1.0)
        
        angles = np.degrees(np.arctan2(yy - center, xx - center)) % 180
        angle_bins = np.linspace(0, 180, 37)
        energy_per_angle = []
        for i in range(len(angle_bins) - 1):
            mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1]) & hf_mask
            energy_per_angle.append(log_mag[mask].sum() if mask.any() else 0)
        
        energy_per_angle = np.array(energy_per_angle)
        if energy_per_angle.max() > 0:
            blur_bin = np.argmin(energy_per_angle)
            blur_angle = (angle_bins[blur_bin] + angle_bins[blur_bin + 1]) / 2
            blur_magnitude = 1.0 - energy_per_angle.min() / max(energy_per_angle.max(), 0.001)
        else:
            blur_angle = 0
            blur_magnitude = 0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_val = gray.mean()
        std_val = gray.std()
        
        dead_pixels = []
        hot_pixels = []
        
        if std_val > 0:
            dead_mask = gray < max(mean_val - 5 * std_val, 1)
            hot_mask = gray > min(mean_val + 5 * std_val, 254)
            
            dead_coords = np.argwhere(dead_mask)
            hot_coords = np.argwhere(hot_mask)
            
            for coord in dead_coords[:20]:
                dead_pixels.append((int(coord[1]), int(coord[0])))
            for coord in hot_coords[:20]:
                hot_pixels.append((int(coord[1]), int(coord[0])))
        
        hf_values = log_mag[hf_mask] if hf_mask.any() else np.array([0])
        noise_level = float(hf_values.std()) if len(hf_values) > 0 else 0.0
        
        block_size = max(fft_size // 8, 1)
        uniformity_scores = []
        for i in range(8):
            for j in range(8):
                y_start = min(i * block_size, fft_size - block_size)
                x_start = min(j * block_size, fft_size - block_size)
                block = log_mag[y_start:y_start+block_size, x_start:x_start+block_size]
                if block.size > 0:
                    uniformity_scores.append(block.mean())
        
        if uniformity_scores and np.mean(uniformity_scores) > 0:
            texture_uniformity = 1.0 - min(np.std(uniformity_scores) / max(np.mean(uniformity_scores), 0.001) * 2, 1.0)
        else:
            texture_uniformity = 0.5
        
        sharpness_score = 1.0 - min(blur_magnitude, 1.0)
        
        return QualityResult(
            focus_score=min(focus_score, 1.0),
            blur_angle=blur_angle,
            blur_magnitude=blur_magnitude,
            dead_pixels=dead_pixels,
            hot_pixels=hot_pixels,
            noise_level=noise_level,
            texture_uniformity=texture_uniformity,
            sharpness_score=sharpness_score
        )
    
    @staticmethod
    def detect_anomalies_ml(fft_result: FFTResult, quality_result: QualityResult) -> Tuple[List[str], float]:
        """ML-based anomaly detection"""
        anomalies = []
        anomaly_score = 0.0
        
        if not SKLEARN_AVAILABLE or not config.ENABLE_ML:
            return anomalies, anomaly_score
        
        try:
            # Extract features
            features = []
            features.append(fft_result.mean_power)
            features.append(fft_result.spectral_entropy)
            features.append(quality_result.focus_score)
            features.append(quality_result.sharpness_score)
            features.append(quality_result.texture_uniformity)
            features.append(quality_result.noise_level)
            features.append(len(fft_result.peaks) / 10.0)
            
            if len(fft_result.peaks) > 0:
                peak_freqs = [p['frequency'] for p in fft_result.peaks[:5]]
                features.extend(peak_freqs)
                features.append(np.mean(peak_freqs) if peak_freqs else 0)
                features.append(np.std(peak_freqs) if len(peak_freqs) > 1 else 0)
            else:
                features.extend([0] * 7)
            
            # Use isolation forest for anomaly detection
            X = np.array(features).reshape(1, -1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            clf = IsolationForest(contamination=0.1, random_state=42)
            pred = clf.fit_predict(X_scaled)
            
            if pred[0] == -1:
                anomalies.append("Isolation forest detected anomaly in spectral features")
                anomaly_score += 0.3
            
            # Additional rule-based anomalies
            if quality_result.focus_score < 0.2:
                anomalies.append("Very poor focus detected")
                anomaly_score += 0.2
            
            if quality_result.noise_level > 0.5:
                anomalies.append("High noise level detected")
                anomaly_score += 0.15
            
            if len(quality_result.dead_pixels) > 10:
                anomalies.append(f"Multiple dead pixels detected: {len(quality_result.dead_pixels)}")
                anomaly_score += 0.15
            
            if quality_result.texture_uniformity < 0.2:
                anomalies.append("Extremely non-uniform texture")
                anomaly_score += 0.1
            
            anomaly_score = min(anomaly_score, 1.0)
            
        except Exception as e:
            warnings.warn(f"ML anomaly detection failed: {e}")
        
        return anomalies, anomaly_score

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class ForensicVisualizer:
    """Generate forensic visualizations"""
    
    @staticmethod
    def render_psd_wheel(log_magnitude: np.ndarray, fft_size: int, 
                         wheel_px: int = 400) -> np.ndarray:
        """Render PSD wheel visualization"""
        yy, xx = np.mgrid[0:wheel_px, 0:wheel_px].astype(np.float32)
        cx = cy = wheel_px / 2.0
        r = wheel_px / 2.0
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx*dx + dy*dy)
        mask = dist <= r
        ang = np.arctan2(dy, dx)
        ang[ang < 0] += 2 * np.pi
        
        freq_r = (dist / r) * (fft_size / 2.0)
        u = np.clip(np.round(fft_size/2 + freq_r * np.cos(ang)).astype(np.int32), 0, fft_size-1)
        v = np.clip(np.round(fft_size/2 + freq_r * np.sin(ang)).astype(np.int32), 0, fft_size-1)
        hue_deg = np.degrees(ang)
        
        power_field = log_magnitude[v, u]
        
        u8 = np.clip(power_field * 255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq = clahe.apply(u8).astype(np.float32) / 255.0
        remapped = np.power(np.clip(eq * 0.6 + power_field * 0.4, 0, 1), 0.45)
        hue = (remapped * 360) % 360
        val = np.power(np.clip(remapped, 0.05, 1.0), 0.7)
        
        hsv = np.zeros((wheel_px, wheel_px, 3), dtype=np.uint8)
        hsv[..., 0] = (hue / 2.0).astype(np.uint8)
        hsv[..., 1] = 230
        hsv[..., 2] = np.clip(val * 255, 10, 255).astype(np.uint8)
        wheel_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        wheel_bgr[~mask] = 0
        
        freq_rings = [(0.25, "f/4"), (0.50, "f/2"), (0.75, "3f/4"), (1.00, "fN")]
        for frac, label in freq_rings:
            radius = int(r * frac)
            if 0 < radius < int(r):
                cv2.circle(wheel_bgr, (int(cx), int(cy)), radius, (255, 255, 255), 1)
                cv2.putText(wheel_bgr, label, (int(cx + radius + 4), int(cy - 4)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        
        return wheel_bgr
    
    @staticmethod
    def render_raw_fft(log_magnitude: np.ndarray, display_px: int = 300) -> np.ndarray:
        """Render raw FFT display"""
        display = (log_magnitude * 255).astype(np.uint8)
        bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        bgr = cv2.resize(bgr, (display_px, display_px), interpolation=cv2.INTER_NEAREST)
        cx = display_px // 2
        cv2.line(bgr, (cx, 0), (cx, display_px), (0, 255, 255), 1)
        cv2.line(bgr, (0, cx), (display_px, cx), (0, 255, 255), 1)
        return bgr
    
    @staticmethod
    def create_forensic_card(image_bgr: np.ndarray, wheel_bgr: np.ndarray,
                             metadata: Dict) -> np.ndarray:
        """Create forensic report card"""
        h, w = image_bgr.shape[:2]
        card_w = 600
        aspect = min(max(h / max(w, 1), 0.4), 2.0)
        frame_h = int(card_w * aspect)
        card_h = frame_h + 80
        
        card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
        card[:] = (10, 14, 20)
        
        frame_resized = cv2.resize(image_bgr, (card_w, frame_h))
        card[:frame_h, :card_w] = frame_resized
        
        wheel_small = cv2.resize(wheel_bgr, (100, 100))
        wx, wy = card_w - 110, 10
        roi = card[wy:wy+100, wx:wx+100]
        card[wy:wy+100, wx:wx+100] = cv2.addWeighted(roi, 0.2, wheel_small, 0.8, 0)
        
        cv2.rectangle(card, (0, frame_h), (card_w, card_h), (5, 10, 18), -1)
        cv2.line(card, (0, frame_h), (card_w, frame_h), (57, 186, 230), 1)
        
        y = frame_h + 20
        cv2.putText(card, f"CASE: {metadata.get('case_id', 'UNKNOWN')}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (57, 186, 230), 1, cv2.LINE_AA)
        cv2.putText(card, f"FILE: {metadata.get('filename', '')[:30]}", (10, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 150, 160), 1, cv2.LINE_AA)
        cv2.putText(card, f"DATE: {metadata.get('timestamp', '')}", (10, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 150, 160), 1, cv2.LINE_AA)
        
        rx = card_w - 200
        cv2.putText(card, f"FFT PWR: {metadata.get('mean_power', 0):.3f}", (rx, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 143, 64), 1, cv2.LINE_AA)
        cv2.putText(card, f"FOCUS: {metadata.get('focus_score', 0):.2f}", (rx, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 143, 64), 1, cv2.LINE_AA)
        cv2.putText(card, f"UNIFORM: {metadata.get('uniformity', 0):.2f}", (rx, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 143, 64), 1, cv2.LINE_AA)
        
        return card
    
    @staticmethod
    def plot_energy_bands(fft_result: FFTResult) -> Figure:
        """Plot energy distribution across frequency bands"""
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0a0e14')
        ax.set_facecolor('#0a0e14')
        
        fft_size = fft_result.fft_size
        log_mag = fft_result.log_magnitude
        center = fft_size // 2
        
        yy, xx = np.ogrid[:fft_size, :fft_size]
        dist = np.sqrt((xx - center)**2 + (yy - center)**2)
        
        bands = [
            ("Low\n(0-25%)", 0, 0.25),
            ("Mid-Low\n(25-40%)", 0.25, 0.40),
            ("Mid-High\n(40-60%)", 0.40, 0.60),
            ("High\n(60%+)", 0.60, 1.0),
        ]
        
        values = []
        labels = []
        for name, lo, hi in bands:
            mask = (dist >= fft_size * lo) & (dist < fft_size * hi)
            values.append(log_mag[mask].mean() if mask.any() else 0)
            labels.append(name)
        
        colors = ['#39bae6', '#7fd962', '#ff8f40', '#f26d78']
        ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel('Mean Log Power', color='#c8ccd4', fontsize=10)
        ax.set_title('Spectral Energy Distribution', color='#39bae6', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#c8ccd4', labelsize=9)
        ax.spines['bottom'].set_color('#1e2a3a')
        ax.spines['top'].set_color('#1e2a3a')
        ax.spines['left'].set_color('#1e2a3a')
        ax.spines['right'].set_color('#1e2a3a')
        ax.grid(axis='y', alpha=0.1, color='white')
        
        plt.tight_layout()
        return fig

# ═══════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class ForensicReportGenerator:
    """Generate professional PDF reports"""
    
    @staticmethod
    def generate_pdf(report: ForensicReport) -> bytes:
        """Generate complete forensic analysis PDF report"""
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=15*mm, bottomMargin=15*mm,
        )
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            'ForensicTitle', parent=styles['Title'],
            fontName='Courier', fontSize=18, textColor=HexColor('#39bae6'),
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            'ForensicHeading', parent=styles['Heading2'],
            fontName='Courier', fontSize=14, textColor=HexColor('#ff8f40'),
            spaceAfter=10, spaceBefore=20,
        ))
        styles.add(ParagraphStyle(
            'ForensicBody', parent=styles['Normal'],
            fontName='Courier', fontSize=9, textColor=HexColor('#c8ccd4'),
            leading=14,
        ))
        styles.add(ParagraphStyle(
            'ForensicMono', parent=styles['Normal'],
            fontName='Courier', fontSize=8, textColor=HexColor('#5c6a7a'),
            leading=12,
        ))
        
        story = []
        
        # Title page
        story.append(Paragraph("SPECTRALEYE FORENSIC ANALYSIS REPORT", styles['ForensicTitle']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Case ID: {report.case_id}", styles['ForensicBody']))
        story.append(Paragraph(f"Analyst: {report.analyst}", styles['ForensicBody']))
        story.append(Paragraph(f"Timestamp: {report.timestamp}", styles['ForensicBody']))
        story.append(Paragraph(f"Source: {report.source_file}", styles['ForensicBody']))
        story.append(Paragraph(f"Analysis Mode: {report.analysis_mode or 'N/A'}", styles['ForensicBody']))
        story.append(Spacer(1, 20))
        
        # Source information
        story.append(Paragraph("SOURCE INFORMATION", styles['ForensicHeading']))
        info_data = [
            ["Property", "Value"],
            ["File", report.source_file],
            ["Type", report.source_type],
            ["Dimensions", f"{report.image_dimensions[0]}×{report.image_dimensions[1]}"],
            ["File Size", f"{report.file_size_bytes:,} bytes"],
            ["MD5 Hash", report.md5_hash],
        ]
        info_table = Table(info_data, colWidths=[80*mm, 80*mm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a2332')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#39bae6')),
            ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#c8ccd4')),
            ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#1e2a3a')),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#0a0e14')),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Spectral analysis
        if report.fft_results:
            story.append(Paragraph("SPECTRAL ANALYSIS RESULTS", styles['ForensicHeading']))
            story.append(Paragraph(f"Mean Spectral Power: {report.fft_results.mean_power:.4f}", styles['ForensicBody']))
            story.append(Paragraph(f"Spectral Entropy: {report.fft_results.spectral_entropy:.4f}", styles['ForensicBody']))
            story.append(Paragraph(f"Computation Time: {report.fft_results.computation_time_ms:.1f}ms", styles['ForensicBody']))
            
            if report.fft_results.peaks:
                peak_data = [["#", "Frequency", "Angle", "Magnitude"]]
                for i, p in enumerate(report.fft_results.peaks[:10]):
                    peak_data.append([
                        str(i+1),
                        f"{p['frequency']:.4f}",
                        f"{p['angle_deg']:.1f}°",
                        f"{p['magnitude']:.4f}",
                    ])
                peak_table = Table(peak_data, colWidths=[20*mm, 45*mm, 45*mm, 50*mm])
                peak_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a2332')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ff8f40')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#c8ccd4')),
                    ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#1e2a3a')),
                ]))
                story.append(peak_table)
        
        story.append(Spacer(1, 20))
        
        # Forgery indicators
        story.append(Paragraph("FORGERY & AUTHENTICATION ANALYSIS", styles['ForensicHeading']))
        
        forgery_data = [
            ["Test", "Result", "Score"],
        ]
        
        if report.deepfake_results:
            forgery_data.append([
                "Deepfake Detection", 
                "⚠ DETECTED" if report.deepfake_results.detected else "✓ CLEAR",
                f"{report.deepfake_results.score:.3f}"
            ])
        else:
            forgery_data.append(["Deepfake Detection", "N/A", "N/A"])
            
        if report.jpeg_results:
            forgery_data.append([
                "JPEG Ghost Artifacts",
                "⚠ DETECTED" if report.jpeg_results.ghost_detected else "✓ NONE",
                f"{report.jpeg_results.ghost_score:.3f}"
            ])
        else:
            forgery_data.append(["JPEG Ghost Artifacts", "N/A", "N/A"])
        
        forgery_data.append([
            "Copy-Move Regions",
            f"⚠ {len(report.copy_move_regions)} FOUND" if report.copy_move_regions else "✓ NONE",
            "N/A"
        ])
        
        if report.anomaly_score > 0:
            forgery_data.append([
                "ML Anomaly Detection",
                "⚠ DETECTED" if report.anomaly_score > config.ANOMALY_THRESHOLD else "✓ CLEAR",
                f"{report.anomaly_score:.3f}"
            ])
        
        forgery_table = Table(forgery_data, colWidths=[50*mm, 50*mm, 60*mm])
        forgery_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a2332')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ff8f40')),
            ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#c8ccd4')),
            ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#1e2a3a')),
        ]))
        story.append(forgery_table)
        story.append(Spacer(1, 20))
        
        # Quality metrics
        if report.quality_results:
            story.append(Paragraph("QUALITY METRICS", styles['ForensicHeading']))
            quality_data = [
                ["Metric", "Value", "Status"],
                ["Focus Score", f"{report.quality_results.focus_score:.3f}", 
                 "✓" if report.quality_results.focus_score > 0.5 else "⚠"],
                ["Sharpness Score", f"{report.quality_results.sharpness_score:.3f}",
                 "✓" if report.quality_results.sharpness_score > 0.5 else "⚠"],
                ["Motion Blur", f"{report.quality_results.blur_magnitude:.3f} @ {report.quality_results.blur_angle:.0f}°",
                 "✓" if report.quality_results.blur_magnitude < 0.3 else "⚠"],
                ["Dead Pixels", str(len(report.quality_results.dead_pixels)), 
                 "⚠" if report.quality_results.dead_pixels else "✓"],
                ["Hot Pixels", str(len(report.quality_results.hot_pixels)),
                 "⚠" if report.quality_results.hot_pixels else "✓"],
                ["Texture Uniformity", f"{report.quality_results.texture_uniformity:.3f}",
                 "✓" if report.quality_results.texture_uniformity > 0.7 else "⚠"],
            ]
            quality_table = Table(quality_data, colWidths=[50*mm, 50*mm, 60*mm])
            quality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a2332')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#39bae6')),
                ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#c8ccd4')),
                ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#1e2a3a')),
            ]))
            story.append(quality_table)
        
        story.append(Spacer(1, 30))
        story.append(Paragraph("— END OF REPORT —", styles['ForensicMono']))
        story.append(Paragraph(f"SpectralEye Forensic v{config.VERSION} | QCAUS Research", styles['ForensicMono']))
        
        doc.build(story)
        buf.seek(0)
        return buf.read()

# ═══════════════════════════════════════════════════════════════════════
# WEB INTERFACE
# ═══════════════════════════════════════════════════════════════════════

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="SpectralEye Forensic",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ────────────────────────────────────────────────────
for key, val in {
    "report": None,
    "processed": False,
    "db": None,
    "case_id": None,
    "analyst": None,
    "filename": None,
    "current_image": None,
    "gpu_available": CUDA_AVAILABLE and config.ENABLE_GPU,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── CSS Styling ──────────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
        /* Base styles */
        [data-testid="stAppViewContainer"] {
            background: #0a0e14;
            color: #c8ccd4;
        }
        [data-testid="stHeader"] {
            background: #0a0e14;
        }
        [data-testid="stSidebar"] {
            background: #11161e;
            border-right: 1px solid #1e2a3a;
        }
        
        /* Typography */
        h1, h2, h3, h4 {
            color: #39bae6 !important;
            font-family: 'SF Mono', 'Consolas', monospace !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }
        h1 { font-size: 1.8em !important; border-bottom: 1px solid #1e2a3a; padding-bottom: 8px; }
        h2 { font-size: 1.4em !important; }
        h3 { font-size: 1.1em !important; color: #ff8f40 !important; }
        p, li, label, div {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
        }
        
        /* Buttons */
        .stButton > button {
            font-family: 'SF Mono', 'Consolas', monospace !important;
            background: #1a2332 !important;
            color: #39bae6 !important;
            border: 1px solid #2a3a4a !important;
            border-radius: 4px !important;
            transition: all 0.2s;
            font-size: 12px !important;
        }
        .stButton > button:hover {
            background: #243044 !important;
            border-color: #39bae6 !important;
            box-shadow: 0 0 8px rgba(57, 186, 230, 0.15);
        }
        .stButton > button.primary {
            background: #ff8f40 !important;
            color: #0a0e14 !important;
            border-color: #ff8f40 !important;
            font-weight: bold !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-family: 'SF Mono', 'Consolas', monospace !important;
            font-size: 22px !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'SF Mono', 'Consolas', monospace !important;
            font-size: 10px !important;
            color: #5c6a7a !important;
        }
        
        /* Status indicators */
        .status-pass { color: #7fd962 !important; }
        .status-warn { color: #ff8f40 !important; }
        .status-fail { color: #f26d78 !important; }
        .status-info { color: #39bae6 !important; }
        
        /* Cards */
        .metric-card {
            background: #11161e;
            border: 1px solid #1e2a3a;
            border-radius: 6px;
            padding: 16px;
            text-align: center;
        }
        .metric-card .value {
            font-size: 26px;
            font-weight: bold;
            color: #39bae6;
            font-family: 'SF Mono', 'Consolas', monospace;
        }
        .metric-card .label {
            font-size: 10px;
            color: #5c6a7a;
            font-family: 'SF Mono', 'Consolas', monospace;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 4px;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #39bae6, #7fd962) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: #0a0e14;
            border-bottom: 1px solid #1e2a3a;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'SF Mono', 'Consolas', monospace !important;
            font-size: 12px !important;
            color: #5c6a7a !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            padding: 8px 16px !important;
        }
        .stTabs [aria-selected="true"] {
            color: #39bae6 !important;
            border-bottom: 2px solid #39bae6 !important;
        }
        
        /* Badge */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-gpu {
            background: #7fd962;
            color: #0a0e14;
        }
        .badge-cpu {
            background: #5c6a7a;
            color: #0a0e14;
        }
        .badge-ml {
            background: #ff8f40;
            color: #0a0e14;
        }
    </style>
    """, unsafe_allow_html=True)

# ─── Helper Functions ──────────────────────────────────────────────────
def render_metric_card(value: str, label: str, status: str = "info"):
    status_class = f"status-{status}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="value {status_class}">{value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_badge(text: str, type: str = "cpu"):
    st.markdown(f'<span class="badge badge-{type}">{text}</span>', unsafe_allow_html=True)

def get_download_link(data_bytes: bytes, filename: str, label: str = "Download", 
                      mime: str = "application/octet-stream"):
    b64 = base64.b64encode(data_bytes).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}" style="color:#39bae6;text-decoration:none;font-family:monospace;">⬇ {label}</a>'

def numpy_to_pil(array_bgr):
    return Image.fromarray(cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB))

def pil_to_bytes(img, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def process_image(image: np.ndarray, fft_size: int, case_id: str, analyst: str,
                  filename: str, use_ml: bool = True,
                  file_bytes: Optional[bytes] = None,
                  analysis_mode: str = "") -> ForensicReport:
    """Process image through complete analysis pipeline"""
    start_time = time.time()

    # Hash and size must reflect the original source bytes (chain-of-custody
    # integrity), not the decoded/re-encoded pixel buffer.
    if file_bytes is not None:
        source_hash = FileHasher.compute_hash(file_bytes)
        source_size = len(file_bytes)
    else:
        source_hash = FileHasher.compute_hash(image.tobytes())
        source_size = int(image.nbytes)

    # Create report
    report = ForensicReport(
        case_id=case_id,
        analyst=analyst,
        timestamp=datetime.now().isoformat(),
        source_file=filename,
        source_type='image',
        image_dimensions=(image.shape[1], image.shape[0]),
        file_size_bytes=source_size,
        md5_hash=source_hash,
        job_id=f"{case_id}_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}",
        analysis_mode=analysis_mode,
        status=AnalysisStatus.PROCESSING
    )
    
    try:
        # Check cache
        cache_key = f"{case_id}_{filename}_{fft_size}_{use_ml}_{source_hash}"
        cached = _analysis_cache.get(cache_key)
        if cached:
            # IMPORTANT: never mutate the cached object in place. The cache
            # stores a reference, so setting .status on it would also
            # silently change every report returned by earlier cache hits
            # (they're the same Python object). Return an independent copy
            # with only the status field overridden.
            return dataclass_replace(cached, status=AnalysisStatus.CACHED)
        
        # Run analyses
        report.fft_results = ForensicAnalyzer.compute_fft(image, fft_size)
        report.deepfake_results = ForensicAnalyzer.detect_deepfake_artifacts(image, fft_size)
        report.jpeg_results = ForensicAnalyzer.detect_jpeg_ghosts(image, fft_size)
        report.quality_results = ForensicAnalyzer.assess_image_quality(image, fft_size)
        
        # ML anomaly detection
        if use_ml and SKLEARN_AVAILABLE:
            anomalies, score = ForensicAnalyzer.detect_anomalies_ml(
                report.fft_results, report.quality_results
            )
            report.anomalies = anomalies
            report.anomaly_score = score
        
        # Generate visualizations
        wheel = ForensicVisualizer.render_psd_wheel(
            report.fft_results.log_magnitude, fft_size, 400
        )
        raw_fft = ForensicVisualizer.render_raw_fft(
            report.fft_results.log_magnitude
        )
        
        # Store visualizations
        wheel_rgb = cv2.cvtColor(wheel, cv2.COLOR_BGR2RGB)
        raw_fft_rgb = cv2.cvtColor(raw_fft, cv2.COLOR_BGR2RGB)
        
        wheel_pil = Image.fromarray(wheel_rgb)
        raw_fft_pil = Image.fromarray(raw_fft_rgb)
        
        buf = BytesIO()
        wheel_pil.save(buf, format='PNG', optimize=True)
        report.psd_wheel = buf.getvalue()
        
        buf = BytesIO()
        raw_fft_pil.save(buf, format='PNG', optimize=True)
        report.raw_fft = buf.getvalue()
        
        # Generate forensic card
        card_metadata = {
            'case_id': report.case_id,
            'filename': report.source_file,
            'timestamp': report.timestamp[:19],
            'mean_power': report.fft_results.mean_power,
            'focus_score': report.quality_results.focus_score,
            'uniformity': report.quality_results.texture_uniformity,
        }
        card = ForensicVisualizer.create_forensic_card(image, wheel, card_metadata)
        card_rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        card_pil = Image.fromarray(card_rgb)
        buf = BytesIO()
        card_pil.save(buf, format='PNG', optimize=True)
        report.forensic_card = buf.getvalue()
        
        # Copy move regions (placeholder)
        report.copy_move_regions = []
        
        # Generate summary
        report.summary = {
            'overall_confidence': 0.0,
            'findings': [],
            'recommendations': []
        }
        
        if report.deepfake_results:
            report.summary['overall_confidence'] += report.deepfake_results.confidence * 0.4
            if report.deepfake_results.detected:
                report.summary['findings'].append({
                    'type': 'deepfake',
                    'severity': 'high' if report.deepfake_results.score > 0.6 else 'medium',
                    'description': f'Deepfake artifacts detected with {report.deepfake_results.confidence:.1%} confidence'
                })
        
        if report.jpeg_results and report.jpeg_results.ghost_detected:
            report.summary['overall_confidence'] += 0.2
            report.summary['findings'].append({
                'type': 'jpeg_ghost',
                'severity': 'medium',
                'description': f'JPEG compression artifacts detected (score: {report.jpeg_results.ghost_score:.2f})'
            })
        
        if report.anomaly_score > config.ANOMALY_THRESHOLD:
            report.summary['overall_confidence'] += report.anomaly_score * 0.2
            report.summary['findings'].append({
                'type': 'anomaly',
                'severity': 'high' if report.anomaly_score > 0.7 else 'medium',
                'description': f'ML anomaly detected (score: {report.anomaly_score:.2f})'
            })
        
        report.summary['overall_confidence'] = min(report.summary['overall_confidence'], 1.0)
        
        if report.summary['overall_confidence'] > 0.7:
            report.summary['recommendations'].append('High confidence in analysis results')
        elif report.summary['overall_confidence'] > 0.4:
            report.summary['recommendations'].append('Additional verification recommended')
        else:
            report.summary['recommendations'].append('Low confidence - consider re-analysis with higher resolution')
        
        # Processing time
        report.processing_time_ms = (time.time() - start_time) * 1000
        report.status = AnalysisStatus.COMPLETED
        
        # Cache results
        _analysis_cache.set(cache_key, report)
        
        # Save to database
        if config.ENABLE_DATABASE:
            db = DatabaseManager()
            db.save_report(report)
        
        return report
        
    except Exception as e:
        report.status = AnalysisStatus.FAILED
        report.errors.append(str(e))
        raise

# ─── Main Application ──────────────────────────────────────────────────
def main():
    load_css()
    
    # Initialize database
    if config.ENABLE_DATABASE:
        st.session_state.db = DatabaseManager()
    
    # GPU status
    gpu_status = "🟢 GPU" if st.session_state.gpu_available else "🟡 CPU"
    ml_status = "🤖 ML" if SKLEARN_AVAILABLE and config.ENABLE_ML else ""
    
    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;padding:8px 0;border-bottom:1px solid #1e2a3a;margin-bottom:20px;">
        <div style="font-size:2em;">🔬</div>
        <div>
            <div style="font-size:1.6em;font-weight:bold;color:#39bae6;font-family:monospace;">SPECTRALEYE FORENSIC</div>
            <div style="font-size:11px;color:#5c6a7a;font-family:monospace;">
                Professional Image & Video Spectral Analysis Platform v{config.VERSION}
                <span style="margin-left:12px;">•</span>
                <span style="margin-left:12px;">{gpu_status}</span>
                {f'<span style="margin-left:12px;">•</span><span style="margin-left:12px;">{ml_status}</span>' if ml_status else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ ANALYSIS CONFIGURATION")
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            [m.value for m in AnalysisMode],
        )
        
        fft_size_label = st.selectbox(
            "FFT Resolution",
            list(config.FFT_SIZES.keys()),
            index=1,
        )
        fft_size = config.FFT_SIZES[fft_size_label]
        
        st.markdown("---")
        st.markdown("### 📂 SOURCE")
        
        source_type = st.radio(
            "Input Source",
            ["📤 Upload File", "📷 Live Camera", "🔗 URL", "📁 Batch Folder"],
        )

        # Only Upload File and URL are actually implemented. The other two
        # options used to be selectable with no handling code behind them,
        # which silently did nothing when the button was clicked.
        source_ready = source_type.startswith("📤") or source_type.startswith("🔗")
        if not source_ready:
            st.caption("⚠ This input source isn't implemented yet. Choose "
                       "'Upload File' or 'URL' to run an analysis.")
        
        uploaded_file = None
        image_url = None
        
        if source_type.startswith("📤"):
            uploaded_file = st.file_uploader(
                "Upload image or video",
                type=["png", "jpg", "jpeg", "tiff", "bmp", "mp4", "avi", "mov", "webp"],
                help=f"Max size: {config.MAX_FILE_SIZE_MB}MB"
            )
        elif source_type.startswith("🔗"):
            image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        
        st.markdown("---")
        st.markdown("### 📋 CASE INFO")
        
        case_id = st.text_input(
            "Case ID", 
            value=f"CASE-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        analyst = st.text_input("Analyst", value="Forensic Analyst")
        
        st.session_state.case_id = case_id
        st.session_state.analyst = analyst
        st.session_state.filename = uploaded_file.name if uploaded_file else image_url or "unknown"
        
        st.markdown("---")
        st.markdown("### 🧠 ML SETTINGS")
        
        use_ml = st.checkbox("Enable ML Detection", value=config.ENABLE_ML and SKLEARN_AVAILABLE)
        
        st.markdown("---")
        
        analyze_button = st.button(
            "🔍 RUN ANALYSIS", use_container_width=True, type="primary",
            disabled=not source_ready,
        )
    
    # Main content — a persistent top-level tab holds the About/User Guide
    # so it's reachable at any time, not just after an analysis has run.
    main_tab, validate_tab, guide_tab = st.tabs(
        ["🔬 ANALYZE", "🧪 VALIDATION & SELF-TEST", "📖 ABOUT & USER GUIDE"]
    )

    with main_tab:
        try:
            if analyze_button:
                with st.spinner("Running forensic analysis pipeline..."):
                    image = None
                    file_bytes = None

                    # Get image from source
                    if uploaded_file is not None:
                        file_bytes = uploaded_file.read()
                        filename = uploaded_file.name

                        FileValidator.validate_file(filename, file_bytes)

                        nparr = np.frombuffer(file_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if image is None:
                            st.error("Unable to decode image. Please check the file format.")
                            return

                        st.session_state.filename = filename

                    elif image_url:
                        try:
                            file_bytes = SSRFProtection.fetch_url_safe(
                                image_url,
                                timeout=config.REQUEST_TIMEOUT_SECONDS,
                                max_size_mb=config.MAX_FILE_SIZE_MB
                            )
                            nparr = np.frombuffer(file_bytes, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if image is None:
                                st.error("Unable to decode image from URL")
                                return
                        except SecurityError as e:
                            st.error(f"Security error: {str(e)}")
                            return

                    if image is None:
                        st.info("Please upload an image or provide a valid URL")
                        return

                    # Process image
                    report = process_image(
                        image, fft_size, case_id, analyst,
                        st.session_state.filename, use_ml,
                        file_bytes=file_bytes, analysis_mode=analysis_mode,
                    )
                    st.session_state.report = report
                    st.session_state.processed = True
                    st.session_state.current_image = image

                    # Display results
                    display_results(report, image)

            elif st.session_state.processed and st.session_state.report:
                display_results(st.session_state.report, st.session_state.current_image)
            else:
                st.info("👈 Configure a case in the sidebar, upload an image (or "
                        "paste a URL), then click **RUN ANALYSIS**. New to the "
                        "tool? See the **ABOUT & USER GUIDE** tab above.")

        except SecurityError as e:
            st.error(f"🔒 Security error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            if config.DEBUG:
                import traceback
                st.code(traceback.format_exc())

    with validate_tab:
        render_validation_tab()

    with guide_tab:
        render_about_guide()

# ─── Validation / Self-Test ─────────────────────────────────────────────
def _gen_natural_test_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Synthetic 'clean/real' negative control: smooth multi-octave noise
    with a natural 1/f-like spectral falloff and no injected periodic
    artifact. Not a real photo — a stand-in with no known artifact."""
    img = np.zeros((size, size), dtype=np.float32)
    for octave, sigma in enumerate([32, 16, 8, 4, 2]):
        layer = rng.normal(0, 1, (size, size)).astype(np.float32)
        layer = cv2.GaussianBlur(layer, (0, 0), sigma)
        img += layer / (octave + 1)
    img -= img.min()
    img /= max(img.max(), 1e-6)
    img *= 255
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def _gen_checkerboard_test_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Synthetic positive control for the deepfake heuristic: a literal
    nearest-neighbor upsampled low-res grid, which is the textbook
    transposed-convolution "checkerboard artifact" (Odena et al., 2016)
    the detector is designed to find. This is a controlled signal-processing
    test case, not a real GAN/diffusion output."""
    factor = int(rng.choice([2, 3, 4, 8]))
    small = (rng.random((size // factor, size // factor)) * 255).astype(np.uint8)
    up = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
    up = cv2.GaussianBlur(up, (3, 3), rng.uniform(0.4, 1.0))
    return cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)

def _run_synthetic_selftest(n_per_class: int, fft_size: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    df_pos, df_neg, jg_pos, jg_neg = [], [], [], []
    for _ in range(n_per_class):
        fake = _gen_checkerboard_test_image(rng, 256)
        real = _gen_natural_test_image(rng, 256)
        df_pos.append(ForensicAnalyzer.detect_deepfake_artifacts(fake, fft_size).score)
        df_neg.append(ForensicAnalyzer.detect_deepfake_artifacts(real, fft_size).score)

        quality = int(rng.choice([15, 25, 35, 50, 65, 80]))
        ok, enc = cv2.imencode('.jpg', real, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpeg_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        jg_pos.append(ForensicAnalyzer.detect_jpeg_ghosts(jpeg_img, fft_size).ghost_score)
        jg_neg.append(ForensicAnalyzer.detect_jpeg_ghosts(real, fft_size).ghost_score)

    df_pos, df_neg = np.array(df_pos), np.array(df_neg)
    jg_pos, jg_neg = np.array(jg_pos), np.array(jg_neg)
    df_thr, jg_thr = config.DEEPFAKE_THRESHOLD, config.JPEG_GHOST_THRESHOLD

    result = {
        "deepfake": {
            "tpr": float((df_pos > df_thr).mean()),
            "fpr": float((df_neg > df_thr).mean()),
            "mean_pos": float(df_pos.mean()), "mean_neg": float(df_neg.mean()),
        },
        "jpeg_ghost": {
            "tpr": float((jg_pos > jg_thr).mean()),
            "fpr": float((jg_neg > jg_thr).mean()),
            "mean_pos": float(jg_pos.mean()), "mean_neg": float(jg_neg.mean()),
        },
    }
    if SKLEARN_AVAILABLE:
        from sklearn.metrics import roc_auc_score
        result["deepfake"]["auc"] = float(roc_auc_score(
            np.r_[np.ones_like(df_pos), np.zeros_like(df_neg)], np.r_[df_pos, df_neg]))
        result["jpeg_ghost"]["auc"] = float(roc_auc_score(
            np.r_[np.ones_like(jg_pos), np.zeros_like(jg_neg)], np.r_[jg_pos, jg_neg]))
    return result

def render_validation_tab():
    st.markdown("""
### 🧪 Validation & Self-Test

Two ways to check what this tool's detectors actually do — neither one is a
substitute for the other.
""")

    with st.expander("① Synthetic self-test — runs instantly, no data needed", expanded=True):
        st.markdown("""
Generates controlled positive/negative test images with a **known, injected**
signal for each detector — a literal nearest-neighbor-upsampled checkerboard
pattern for the deepfake heuristic, and genuine JPEG recompression for the
ghost heuristic — against synthetic clean negatives with no such signal.

This answers "**does the code detect the specific artifact it claims to
detect, at the configured threshold?**" It does **not** answer "does this
detect real deepfakes or real photos" — for that, use section ② below with
actual labeled images.
""")
        n_per_class = st.slider("Test images per class", 10, 100, 40, 10, key="selftest_n")
        if st.button("▶ Run synthetic self-test", key="run_selftest"):
            with st.spinner("Generating synthetic test corpus and scoring..."):
                res = _run_synthetic_selftest(n_per_class, 256, seed=int(time.time()))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Deepfake / upsampling-artifact heuristic**")
                render_metric_card(f"{res['deepfake']['tpr']:.0%}", "TPR on synthetic fakes",
                                  "pass" if res['deepfake']['tpr'] > 0.7 else "warn")
                render_metric_card(f"{res['deepfake']['fpr']:.0%}", "FPR on synthetic reals",
                                  "pass" if res['deepfake']['fpr'] < 0.15 else "warn")
                if 'auc' in res['deepfake']:
                    st.caption(f"AUC: {res['deepfake']['auc']:.3f}")
            with c2:
                st.markdown("**JPEG-ghost / recompression heuristic**")
                render_metric_card(f"{res['jpeg_ghost']['tpr']:.0%}", "TPR on recompressed",
                                  "pass" if res['jpeg_ghost']['tpr'] > 0.5 else "warn")
                render_metric_card(f"{res['jpeg_ghost']['fpr']:.0%}", "FPR on fresh images",
                                  "pass" if res['jpeg_ghost']['fpr'] < 0.35 else "warn")
                if 'auc' in res['jpeg_ghost']:
                    st.caption(f"AUC: {res['jpeg_ghost']['auc']:.3f}")
            st.caption("Reference (held-out synthetic test set, n=60/class): deepfake "
                       "TPR≈88% / FPR≈7% (AUC≈0.92); JPEG-ghost TPR≈63% / FPR≈32% "
                       "(AUC≈0.69). Your numbers will vary run to run — synthetic "
                       "images are freshly randomized each time.")

    with st.expander("② Validate against your own labeled images — the real test", expanded=False):
        st.markdown("""
Upload images you *know* are real/unmanipulated and images you *know* are
AI-generated or manipulated. This computes actual accuracy against ground
truth you supply, instead of synthetic stand-ins — this is the only way to
get a number that means something for your specific use case.
""")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            real_files = st.file_uploader("Known REAL / unmanipulated images", type=["png","jpg","jpeg","bmp","webp"],
                                          accept_multiple_files=True, key="val_real")
        with vcol2:
            fake_files = st.file_uploader("Known FAKE / AI-generated / manipulated images", type=["png","jpg","jpeg","bmp","webp"],
                                          accept_multiple_files=True, key="val_fake")

        if st.button("▶ Run labeled validation", key="run_labeled_val"):
            if not real_files and not fake_files:
                st.warning("Upload at least one image in each category.")
            else:
                rows = []
                with st.spinner(f"Scoring {len(real_files)+len(fake_files)} uploaded images..."):
                    for f, label in [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]:
                        try:
                            data = f.read()
                            nparr = np.frombuffer(data, np.uint8)
                            im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if im is None:
                                continue
                            df = ForensicAnalyzer.detect_deepfake_artifacts(im, 256)
                            rows.append({"filename": f.name, "label": label,
                                       "deepfake_score": df.score, "predicted": int(df.detected)})
                        except Exception as e:
                            st.caption(f"Skipped {f.name}: {e}")

                if not rows:
                    st.error("No images could be scored.")
                else:
                    labels = np.array([r["label"] for r in rows])
                    preds = np.array([r["predicted"] for r in rows])
                    scores = np.array([r["deepfake_score"] for r in rows])

                    tp = int(((preds == 1) & (labels == 1)).sum())
                    fp = int(((preds == 1) & (labels == 0)).sum())
                    tn = int(((preds == 0) & (labels == 0)).sum())
                    fn = int(((preds == 0) & (labels == 1)).sum())
                    accuracy = (tp + tn) / max(len(rows), 1)
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)

                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1: render_metric_card(f"{accuracy:.0%}", "Accuracy", "info")
                    with mcol2: render_metric_card(f"{precision:.0%}", "Precision", "info")
                    with mcol3: render_metric_card(f"{recall:.0%}", "Recall (TPR)", "info")
                    with mcol4: render_metric_card(f"{fp/max(fp+tn,1):.0%}", "False Positive Rate", "info")

                    if SKLEARN_AVAILABLE and len(set(labels.tolist())) > 1:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(labels, scores)
                        st.caption(f"AUC: {auc:.3f}")

                    st.markdown(f"**Confusion matrix** — TP={tp} FP={fp} TN={tn} FN={fn}")
                    st.dataframe(rows, use_container_width=True)
    st.caption("⚠ This tool is a spectral/compression forensics utility, not a "
               "trained AI-content classifier. See the About & User Guide tab "
               "for scope and known limitations before relying on either "
               "validation result for a real decision.")

# ─── About / User Guide ─────────────────────────────────────────────────
def render_about_guide():
    """Static About & User Guide content — always available, independent
    of whether an analysis has been run yet."""

    st.markdown(f"""
## 🔬 About SpectralEye Forensic

**SpectralEye Forensic** is a frequency-domain (FFT-based) image analysis
tool for authentication, quality assessment, and anomaly screening. It looks
at an image's *spectral* signature — how its energy is distributed across
spatial frequencies — to surface patterns associated with resampling,
JPEG recompression, GAN upsampling artifacts, blur, and sensor noise.

**Version:** {config.VERSION} &nbsp;•&nbsp; **Author:** Tony E. Ford | QCAUS Research

---

### ⚠️ Important limitations — please read

- **This is not a general AI-content detector, and should not be pitched or
  used as one.** It targets two specific, narrow signal-processing artifacts:
  (1) periodic "checkerboard" spectral energy left by *transposed-convolution*
  upsampling in older GAN architectures, and (2) residual 8×8-DCT block-grid
  periodicity from a prior JPEG compression pass. **Modern diffusion models
  (Stable Diffusion, Midjourney, DALL·E, Flux, and most current-generation
  image generators) do not use transposed-convolution upsampling and do not
  reliably produce artifact (1) at all.** A clean result here says nothing
  about whether an image came from a current AI generator.
- **Both heuristics have been validated only against synthetic, controlled
  test images** (see the 🧪 Validation & Self-Test tab) — not against real
  photographs or real AI-generated images from named generators. On a
  held-out synthetic test set: the deepfake/upsampling heuristic reached
  ≈88% true-positive rate at ≈7% false-positive rate (AUC≈0.92) for the
  *specific artifact it targets*; the JPEG-ghost heuristic reached only
  ≈63% true-positive / ≈32% false-positive (AUC≈0.69) — a real but modest
  signal, weaker than the deepfake heuristic, and it degrades further at
  high JPEG quality (≥90) where the block-grid signal is naturally faint.
  **Neither number tells you how the tool performs on real, unlabeled
  images** — run the labeled-upload validation in that tab with your own
  known real/fake images before trusting either score for a real decision.
- **This tool does not produce legal-grade "authentic/fake" verdicts.**
  Every score is a heuristic signal, not proof. Treat all findings as
  investigative leads that need corroboration, not conclusions.
- **The ML anomaly detector fits an Isolation Forest on a single image's
  feature vector at analysis time.** With one sample there is no real
  population to compare against, so its "contamination" estimate is a soft
  heuristic overlay on the rule-based checks, not a statistically robust
  outlier test. Don't weight it as strongly as the deterministic FFT metrics.
- **Video files are not currently analyzed.** The uploader accepts common
  video extensions for forward-compatibility, but only the still-image
  pipeline is implemented — a video upload will fail to decode.
- **"Live Camera" and "Batch Folder" input modes are not implemented yet**
  and are disabled in the sidebar until they are.

---

### 📖 How to use it

1. **Configure the case** — in the sidebar, set an *Analysis Mode* (informational
   — currently all modes run the same full pipeline), pick an *FFT Resolution*
   (higher = more detail, slower), and fill in a *Case ID* / *Analyst* name for
   the report.
2. **Provide a source image** — either *Upload File* (PNG/JPG/TIFF/BMP/WEBP) or
   paste an *Image URL*. Uploaded files are validated by extension, size limit,
   and file-signature (magic bytes) before decoding.
3. **Click RUN ANALYSIS.** The pipeline computes:
   - An FFT power spectrum and its dominant frequency peaks
   - A deepfake/GAN-artifact heuristic score
   - A JPEG recompression ("ghost") heuristic score
   - Focus, blur direction/magnitude, noise, and pixel-defect quality metrics
   - Optionally, an ML anomaly overlay (see limitation above)
4. **Read the results across the tabs:**
   - **Overview** — source image, PSD wheel, raw FFT, and top frequency peaks
   - **Forgery** — deepfake and JPEG-ghost details, plus any ML anomalies
   - **Quality** — focus/sharpness/blur/noise/dead-pixel metrics
   - **Report** — the full structured JSON of everything computed
   - **Export** — download the forensic card (PNG), PDF report, JSON data,
     or a ZIP bundling all of it plus a CSV of frequency peaks
5. **Chain of custody:** the MD5 hash and file size recorded in the report
   are computed from the *original uploaded/fetched bytes*, not the decoded
   pixel buffer, so they'll match the source file's own hash if you need to
   verify integrity independently.
6. **Before trusting a Forgery result, check the 🧪 Validation & Self-Test
   tab** — run the synthetic self-test to confirm the detectors are firing
   as expected in your deployment, and ideally the labeled-upload validation
   with images relevant to your actual use case.

---

### 🧾 What the key metrics mean

| Metric | What it measures | Higher generally means |
|---|---|---|
| **Deepfake Score** | Ring-averaged spectral energy at known upsampling-artifact frequencies (1/2, 1/3, 1/4, 1/8 harmonics) vs. local background, blended with a high-frequency energy ratio | More transposed-conv-upsampling-like spectral signature (see limitations — does not cover diffusion models) |
| **JPEG Ghost Score** | Sharp comb-spike energy at 8×8-DCT block-grid harmonics along the horizontal/vertical frequency axes, vs. smoothly-interpolated neighboring points on the same axis | More evidence of a prior JPEG compression pass (moderate-confidence signal, see limitations) |
| **Focus Score** | Share of spectral energy in high frequencies | Sharper / more in-focus image |
| **Blur Magnitude / Angle** | Directional dip in high-frequency energy | Stronger, more directional motion blur |
| **Texture Uniformity** | Spatial evenness of spectral power across an 8×8 grid | More uniform texture across the frame |
| **Anomaly Score** | Isolation Forest + rule-based checks on the above (see limitation) | More features falling outside expected ranges |

---

### 🔒 Data handling

- Analysis runs locally in this session; results are cached in-memory
  (LRU, {config.CACHE_TTL_SECONDS//60} min TTL) and optionally persisted to a
  local SQLite database (`{config.DB_PATH}`) if database storage is enabled.
- The *Image URL* option fetches only `image/*` content types and blocks
  requests to localhost, private/loopback/multicast IP ranges, and
  non-http(s) schemes (SSRF protection).
- Nothing is sent to a third-party service — all computation (FFT, quality
  metrics, PDF/report generation) happens in this process.

---

### 🧭 Troubleshooting

- **"Unable to decode image"** — the file extension didn't match its actual
  content, or the file is corrupted / not a supported still-image format.
- **RUN ANALYSIS is disabled** — you have "Live Camera" or "Batch Folder"
  selected as the input source; switch to "Upload File" or "URL".
- **Security error on URL fetch** — the URL pointed at a non-image
  content-type, a blocked host/IP, or exceeded the size limit.
""")

# ─── Results Display ──────────────────────────────────────────────────
def display_results(report: ForensicReport, image: Optional[np.ndarray] = None):
    """Display analysis results"""
    
    st.markdown("---")
    st.markdown("## 📊 ANALYSIS RESULTS")
    
    # Status
    if report.status == AnalysisStatus.COMPLETED:
        st.success("✅ Analysis completed successfully")
    elif report.status == AnalysisStatus.CACHED:
        st.info("📦 Results retrieved from cache")
    elif report.status == AnalysisStatus.FAILED:
        st.error(f"❌ Analysis failed: {', '.join(report.errors)}")
    
    # Confidence
    if report.summary:
        confidence = report.summary.get('overall_confidence', 0)
        st.markdown(f"""
        <div style="background:#11161e;border:1px solid #1e2a3a;border-radius:8px;padding:12px 16px;margin:8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#c8ccd4;">Overall Confidence</span>
                <span style="color:{'#7fd962' if confidence > 0.7 else '#ff8f40' if confidence > 0.4 else '#f26d78'};font-weight:bold;font-size:18px;">
                    {confidence:.1%}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card(f"{report.image_dimensions[0]}×{report.image_dimensions[1]}", "Dimensions", "info")
    with m2:
        render_metric_card(f"{report.fft_results.mean_power:.3f}" if report.fft_results else "N/A", 
                          "Mean PSD Power", "info")
    with m3:
        focus = report.quality_results.focus_score if report.quality_results else 0
        status = "pass" if focus > 0.5 else "warn"
        render_metric_card(f"{focus:.2f}", "Focus Score", status)
    with m4:
        deepfake = report.deepfake_results.score if report.deepfake_results else 0
        status = "fail" if deepfake > 0.4 else "pass"
        render_metric_card(f"{deepfake:.2f}", "Deepfake Score", status)
    with m5:
        ghost = report.jpeg_results.ghost_detected if report.jpeg_results else False
        status = "fail" if ghost else "pass"
        render_metric_card("DETECTED" if ghost else "CLEAR", "JPEG Ghosts", status)
    
    # Processing info
    st.caption(f"Processing time: {report.processing_time_ms:.0f}ms | "
               f"GPU: {'Yes' if st.session_state.gpu_available else 'No'} | "
               f"ML: {'Yes' if report.anomaly_score > 0 else 'No'} | "
               f"Mode: {report.analysis_mode or 'N/A'} | "
               f"Job: {report.job_id}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Forgery", "📐 Quality", "📋 Report", "💾 Export"
    ])
    
    with tab1:
        if image is not None:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="Source Image", use_container_width=True)
            with col2:
                if report.psd_wheel:
                    st.image(report.psd_wheel, caption="PSD Wheel", use_container_width=True)
            with col3:
                if report.raw_fft:
                    st.image(report.raw_fft, caption="Raw FFT", use_container_width=True)
        else:
            st.info("Source image preview unavailable for this session (report only).")
        
        if report.fft_results and report.fft_results.peaks:
            st.markdown("#### Dominant Frequency Peaks")
            peaks_df = []
            for i, p in enumerate(report.fft_results.peaks[:8]):
                peaks_df.append({
                    "#": i+1,
                    "Spatial Freq": f"{p['frequency']:.4f}",
                    "Angle": f"{p['angle_deg']:.1f}°",
                    "Magnitude": f"{p['magnitude']:.4f}",
                })
            st.dataframe(peaks_df, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔍 Forgery & Authentication Analysis")
        
        if report.deepfake_results:
            col1, col2 = st.columns(2)
            with col1:
                df = report.deepfake_results
                st.markdown(f"""
                #### Deepfake Detection
                - **Score:** {df.score:.4f} {'⚠' if df.detected else '✓'}
                - **Confidence:** {df.confidence:.2%}
                - **Upsample Artifacts:** {df.upsample_artifact_hits} hits
                - **Angle Clustering:** {df.angle_cluster_score:.3f}
                - **HF Anomaly:** {df.hf_anomaly_score:.3f}
                
                **Verdict:** {'⚠ POTENTIAL DEEPFAKE' if df.detected else '✓ No GAN artifacts detected'}
                """)
                
                if report.fft_results:
                    fig = ForensicVisualizer.plot_energy_bands(report.fft_results)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                if report.jpeg_results:
                    jr = report.jpeg_results
                    st.markdown(f"""
                    #### JPEG Compression Ghost Analysis
                    - **Ghost Score:** {jr.ghost_score:.3f}
                    - **Ghost Detected:** {'⚠ YES' if jr.ghost_detected else '✓ NO'}
                    - **Compression Estimate:** {jr.compression_estimate if jr.compression_estimate else 'N/A'}
                    """)
                    
                    if jr.peaks:
                        st.markdown("**JPEG Peak Frequencies:**")
                        for jp in jr.peaks[:5]:
                            st.markdown(f"- k={jp['harmonic_k']}/8 (f={jp['target_freq']:.3f}, "
                                      f"{jp['direction']}) — spike ratio {jp['spike_ratio']:.2f}×")
                
                if report.anomaly_score > 0:
                    st.markdown(f"""
                    #### ML Anomaly Detection
                    - **Anomaly Score:** {report.anomaly_score:.3f}
                    - **Detected Anomalies:** {len(report.anomalies)}
                    
                    **Status:** {'⚠ Anomalies detected' if report.anomaly_score > config.ANOMALY_THRESHOLD else '✓ No significant anomalies'}
                    """)
                    if report.anomalies:
                        st.markdown("**Anomalies:**")
                        for a in report.anomalies[:5]:
                            st.markdown(f"- {a}")
    
    with tab3:
        st.markdown("### 📐 Quality Assessment Metrics")
        
        if report.quality_results:
            q = report.quality_results
            
            quality_score = (q.focus_score + q.sharpness_score + (1 - q.blur_magnitude) + q.texture_uniformity) / 4
            quality_color = "#7fd962" if quality_score > 0.6 else "#ff8f40" if quality_score > 0.3 else "#f26d78"
            
            st.markdown(f"""
            <div style="text-align:center;padding:20px;background:#11161e;border:1px solid #1e2a3a;border-radius:8px;margin:8px 0;">
                <div style="font-size:14px;color:#5c6a7a;text-transform:uppercase;letter-spacing:0.05em;">Overall Image Quality</div>
                <div style="font-size:48px;font-weight:bold;color:{quality_color};">
                    {quality_score:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            qc1, qc2, qc3, qc4 = st.columns(4)
            with qc1:
                render_metric_card(f"{q.focus_score:.3f}", "Focus Score", 
                                  "pass" if q.focus_score > 0.5 else "warn")
            with qc2:
                render_metric_card(f"{q.sharpness_score:.3f}", "Sharpness",
                                  "pass" if q.sharpness_score > 0.5 else "warn")
            with qc3:
                render_metric_card(f"{q.blur_magnitude:.3f}", "Blur Magnitude",
                                  "pass" if q.blur_magnitude < 0.3 else "warn")
            with qc4:
                render_metric_card(f"{q.texture_uniformity:.3f}", "Texture Uniformity",
                                  "pass" if q.texture_uniformity > 0.7 else "warn")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                #### Detailed Metrics
                | Metric | Value |
                |--------|-------|
                | Blur Direction | {q.blur_angle:.1f}° |
                | Noise Level | {q.noise_level:.4f} |
                | Dead Pixels | {len(q.dead_pixels)} |
                | Hot Pixels | {len(q.hot_pixels)} |
                """)
            
            with col2:
                if q.dead_pixels:
                    st.markdown("**Dead Pixel Coordinates:**")
                    st.text("\n".join([f"  ({x}, {y})" for x, y in q.dead_pixels[:5]]))
                if q.hot_pixels:
                    st.markdown("**Hot Pixel Coordinates:**")
                    st.text("\n".join([f"  ({x}, {y})" for x, y in q.hot_pixels[:5]]))
    
    with tab4:
        st.markdown("### 📋 Complete Forensic Report")
        
        # Summary
        if report.summary:
            st.markdown("#### Summary")
            st.json(report.summary)
        
        # Full report
        st.markdown("#### Full Data")
        st.json(report.to_dict())
    
    with tab5:
        st.markdown("### 💾 Export Options")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if report.forensic_card:
                st.download_button(
                    "📷 Forensic Card",
                    data=report.forensic_card,
                    file_name=f"{report.case_id}_card.png",
                    mime="image/png",
                    use_container_width=True
                )
        with col2:
            pdf_bytes = ForensicReportGenerator.generate_pdf(report)
            st.download_button(
                "📄 PDF Report",
                data=pdf_bytes,
                file_name=f"{report.case_id}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col3:
            st.download_button(
                "📊 JSON Data",
                data=json.dumps(report.to_dict(), indent=2),
                file_name=f"{report.case_id}_data.json",
                mime="application/json",
                use_container_width=True
            )
        with col4:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                if report.forensic_card:
                    zf.writestr(f"{report.case_id}_card.png", report.forensic_card)
                zf.writestr(f"{report.case_id}_report.pdf", pdf_bytes)
                zf.writestr(f"{report.case_id}_data.json", 
                           json.dumps(report.to_dict(), indent=2))
                if report.fft_results and report.fft_results.peaks:
                    csv_data = "frequency,angle_deg,magnitude\n"
                    for p in report.fft_results.peaks:
                        csv_data += f"{p['frequency']},{p['angle_deg']},{p['magnitude']}\n"
                    zf.writestr(f"{report.case_id}_peaks.csv", csv_data)
            zip_buf.seek(0)
            st.download_button(
                "📦 Complete ZIP",
                data=zip_buf.read(),
                file_name=f"{report.case_id}_complete.zip",
                mime="application/zip",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
