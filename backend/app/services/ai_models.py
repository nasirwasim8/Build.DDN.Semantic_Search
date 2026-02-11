"""
AI Models service for multimodal analysis.
CLIP for semantic embeddings, BLIP for captioning, ViT for classification.
"""
import os
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Check for GPU
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

# Initialize logger
logger = logging.getLogger(__name__)

# Check for transformers
try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ImageAnalyzer:
    """Image analysis using CLIP and BLIP models."""

    def __init__(self):
        self.models_loaded = False
        self.device = DEVICE

        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            print("Warning: Transformers/PyTorch not available. Image analysis disabled.")
            return

        try:
            print(f"Loading AI models on {self.device}...")

            # BLIP for image captioning
            print("  - Loading BLIP for captioning...")
            self.caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)

            # CLIP for semantic search
            print("  - Loading CLIP for semantic search...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

            # Scene classifier
            print("  - Loading scene classifier...")
            self.scene_classifier = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=0 if self.device == "cuda" else -1
            )

            self.models_loaded = True
            print("All image models loaded successfully")

        except Exception as e:
            print(f"Warning: Model loading error: {e}")
            self.models_loaded = False

    def generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using BLIP."""
        if not self.models_loaded:
            return "AI models not loaded"

        try:
            inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
            outputs = self.caption_model.generate(**inputs, max_new_tokens=50)
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Caption error: {e}"

    def classify_scene(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Classify image scene using ViT."""
        if not self.models_loaded:
            return []

        try:
            results = self.scene_classifier(image, top_k=top_k)
            return results
        except Exception as e:
            print(f"Scene classification error: {e}")
            return []

    def compute_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP image embedding for semantic search."""
        if not self.models_loaded:
            return np.array([])

        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embedding = F.normalize(image_features, p=2, dim=1)

            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"CLIP embedding error: {e}")
            return np.array([])

    def compute_text_embedding(self, text: str) -> np.ndarray:
        """Generate CLIP text embedding for query matching."""
        if not self.models_loaded:
            return np.array([])

        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                embedding = F.normalize(text_features, p=2, dim=1)

            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Text embedding error: {e}")
            return np.array([])

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity between image and text."""
        if not self.models_loaded:
            return 0.0

        try:
            inputs = self.clip_processor(
                text=[text], images=image, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
                similarity = (image_embeds @ text_embeds.T).squeeze().item()
                # Convert from [-1, 1] to [0, 1]
                similarity = (similarity + 1) / 2

            return similarity
        except Exception as e:
            print(f"Similarity error: {e}")
            return 0.0

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive image analysis."""
        try:
            width, height = image.size
            caption = self.generate_caption(image)
            scene_results = self.classify_scene(image)
            objects = ', '.join([r['label'] for r in scene_results[:3]]) if scene_results else 'unknown'
            embedding = self.compute_clip_embedding(image)

            return {
                'caption': caption,
                'width': width,
                'height': height,
                'aspect_ratio': round(width / height, 2),
                'scene_classification': scene_results,
                'detected_objects': objects,
                'dominant_scene': scene_results[0]['label'] if scene_results else 'unknown',
                'embedding_dims': len(embedding) if len(embedding) > 0 else 0
            }
        except Exception as e:
            return {'caption': 'Analysis error', 'error': str(e)}
    
    def generate_embeddings_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Generate CLIP embeddings for a batch of images (efficient for GPU).
        
        Args:
            images: List of PIL images
            
        Returns:
            List of embedding arrays
        """
        if not self.models_loaded or not images:
            return []
        
        try:
            # Process all images in one batch
            inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embeddings = F.normalize(image_features, p=2, dim=1)
            
            # Convert to numpy arrays
            embeddings_np = embeddings.cpu().numpy()
            return [emb for emb in embeddings_np]
            
        except Exception as e:
            logger.error(f"Batch embeddings error: {e}")
            return [np.array([]) for _ in images]
    
    def generate_captions_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Generate captions for a batch of images (efficient for GPU).
        
        Args:
            images: List of PIL images
            
        Returns:
            List of caption strings
        """
        if not self.models_loaded or not images:
            return []
        
        try:
            # Process all images in one batch
            inputs = self.caption_processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.caption_model.generate(**inputs, max_new_tokens=50)
            
            # Decode each caption
            captions = [
                self.caption_processor.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return captions
            
        except Exception as e:
            logger.error(f"Batch captions error: {e}")
            return ["Caption error" for _ in images]



class VideoAnalyzer:
    """Video analysis with frame extraction and AI processing."""

    def __init__(self, image_analyzer: ImageAnalyzer):
        self.image_analyzer = image_analyzer

        try:
            import cv2
            self.cv2 = cv2
            self.cv2_available = True
        except ImportError:
            self.cv2_available = False
            print("Warning: OpenCV not available. Video analysis disabled.")

    def extract_frames(self, video_path: str, num_frames: int = 10) -> List[Image.Image]:
        """Extract key frames from video."""
        if not self.cv2_available:
            return []

        frames = []
        try:
            cap = self.cv2.VideoCapture(video_path)
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                return frames

            # Calculate frame indices to extract
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            for idx in indices:
                cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))

            cap.release()
        except Exception as e:
            print(f"Frame extraction error: {e}")

        return frames

    def get_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata."""
        if not self.cv2_available:
            return {}

        try:
            cap = self.cv2.VideoCapture(video_path)
            fps = cap.get(self.cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))

            metadata = {
                'duration_seconds': total_frames / fps if fps > 0 else 0,
                'fps': fps,
                'width': int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)),
                'total_frames': total_frames
            }
            cap.release()
            return metadata
        except Exception as e:
            print(f"Video metadata error: {e}")
            return {}

    def analyze(self, video_path: str, num_frames: int = 8) -> Dict[str, Any]:
        """Comprehensive video analysis."""
        frames = self.extract_frames(video_path, num_frames)
        metadata = self.get_video_metadata(video_path)

        frame_analyses = []
        all_captions = []
        all_objects = set()

        for i, frame in enumerate(frames):
            analysis = self.image_analyzer.analyze(frame)
            frame_analyses.append({
                'frame_index': i,
                'caption': analysis.get('caption', ''),
                'objects': analysis.get('detected_objects', '')
            })
            all_captions.append(analysis.get('caption', ''))
            for obj in analysis.get('detected_objects', '').split(', '):
                if obj:
                    all_objects.add(obj)

        # Generate summary from captions
        summary = f"Video contains: {', '.join(list(all_objects)[:10])}"
        if all_captions:
            summary += f". Key scenes: {'; '.join(all_captions[:3])}"

        return {
            **metadata,
            'summary': summary,
            'frame_analyses': frame_analyses,
            'detected_objects': ', '.join(list(all_objects)[:15]),
            'key_tags': list(all_objects)[:10]
        }
    
    def search_frames_semantic(self, video_path: str, query: str, threshold: float = 0.20) -> List[Dict]:
        """
        Search video frames using semantic similarity with CLIP.
        
        Args:
            video_path: Path to video file
            query: Semantic search query
            threshold: Minimum CLIP similarity score (0.0-1.0)
            
        Returns:
            List of matching frames with metadata
        """
        import base64
        from io import BytesIO
        
        try:
            # Extract keyframes (20 frames)
            frames = self.extract_frames(video_path, num_frames=20)
            
            if not frames:
                return []
            
            # Get video metadata for timestamps
            metadata = self.get_video_metadata(video_path)
            fps = metadata.get('fps', 30)
            total_frames = metadata.get('total_frames', len(frames))
            
            matching_frames = []
            
            for idx, frame_image in enumerate(frames):
                # Calculate actual frame index and timestamp
                frame_index = int((idx / len(frames)) * total_frames)
                timestamp = frame_index / fps if fps > 0 else 0
                
                # Generate caption using image analyzer
                caption = "Frame content"
                try:
                    caption_result = self.image_analyzer.generate_caption(frame_image)
                    caption = caption_result if caption_result else "Frame content"
                except:
                    pass
                
                # Compute CLIP similarity
                clip_score = 0.0
                try:
                    clip_score = self.image_analyzer.compute_similarity(frame_image, query)
                except Exception as e:
                    print(f"CLIP scoring error: {e}")
                
                # Apply threshold filter
                if clip_score >= threshold:
                    # Convert frame to base64 for transmission
                    buffered = BytesIO()
                    frame_image.save(buffered, format="JPEG", quality=85)
                    frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    matching_frames.append({
                        'frame_index': frame_index,
                        'timestamp': round(timestamp, 2),
                        'caption': caption,
                        'clip_score': round(clip_score, 3),
                        'frame_base64': frame_base64
                    })
            
            # Sort by CLIP score descending
            matching_frames.sort(key=lambda x: x['clip_score'], reverse=True)
            
            return matching_frames
            
        except Exception as e:
            print(f"Error in semantic frame search: {e}")
            return []



class DocumentAnalyzer:
    """Document analysis for PDFs and Word documents."""

    def __init__(self):
        self.summarizer = None
        self.ner_pipeline = None

        # Check for document processing libraries
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
            self.pdf_available = True
        except ImportError:
            self.pdf_available = False

        try:
            import docx
            self.docx = docx
            self.docx_available = True
        except ImportError:
            self.docx_available = False

        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                print("Loading document analysis models...")
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if DEVICE == "cuda" else -1
                )
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    device=0 if DEVICE == "cuda" else -1,
                    aggregation_strategy="simple"
                )
                print("Document analysis models loaded")
            except Exception as e:
                print(f"Document model loading error: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        if not self.pdf_available:
            return ""

        try:
            text = ""
            with open(pdf_path, 'rb') as f:
                reader = self.PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from Word document."""
        if not self.docx_available:
            return ""

        try:
            doc = self.docx.Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return ""

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Generate summary of text."""
        if not self.summarizer or not text:
            return ""

        try:
            # Truncate if too long
            if len(text) > 1024:
                text = text[:1024]

            result = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return ""

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        if not self.ner_pipeline or not text:
            return []

        try:
            # Process in chunks if text is long
            if len(text) > 512:
                text = text[:512]

            entities = self.ner_pipeline(text)
            return [{'entity': e['entity_group'], 'word': e['word'], 'score': e['score']}
                    for e in entities]
        except Exception as e:
            print(f"NER error: {e}")
            return []

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive document analysis."""
        from pathlib import Path
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
        else:
            # Try reading as plain text
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except:
                text = ""

        if not text:
            return {'error': 'Could not extract text from document'}

        summary = self.summarize(text)
        entities = self.extract_entities(text)

        # Extract key terms
        entity_words = [e['word'] for e in entities[:10]]

        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'summary': summary,
            'entities': entities,
            'key_terms': ', '.join(entity_words),
            'preview': text[:500] + '...' if len(text) > 500 else text
        }


# Global instances - lazily initialized
_image_analyzer: Optional[ImageAnalyzer] = None
_video_analyzer: Optional[VideoAnalyzer] = None
_document_analyzer: Optional[DocumentAnalyzer] = None


def get_image_analyzer() -> ImageAnalyzer:
    """Get or create image analyzer instance."""
    global _image_analyzer
    if _image_analyzer is None:
        _image_analyzer = ImageAnalyzer()
    return _image_analyzer


def get_video_analyzer() -> VideoAnalyzer:
    """Get or create video analyzer instance."""
    global _video_analyzer
    if _video_analyzer is None:
        _video_analyzer = VideoAnalyzer(get_image_analyzer())
    return _video_analyzer


def get_document_analyzer() -> DocumentAnalyzer:
    """Get or create document analyzer instance."""
    global _document_analyzer
    if _document_analyzer is None:
        _document_analyzer = DocumentAnalyzer()
    return _document_analyzer
