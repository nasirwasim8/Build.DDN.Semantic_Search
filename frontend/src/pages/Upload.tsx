import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Upload, Image, Video, FileText, Loader2, CheckCircle, X } from 'lucide-react'
import toast from 'react-hot-toast'
import { api, ImageUploadResponse, VideoUploadResponse, DocumentUploadResponse } from '../services/api'

type UploadTab = 'image' | 'video' | 'document'
type UploadResult = ImageUploadResponse | VideoUploadResponse | DocumentUploadResponse

export default function UploadPage() {
  const [activeTab, setActiveTab] = useState<UploadTab>('image')
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [customCaption, setCustomCaption] = useState('')
  const [customSummary, setCustomSummary] = useState('')
  const [customTags, setCustomTags] = useState('')
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)

  const uploadImageMutation = useMutation({
    mutationFn: (file: File) => api.uploadImage(file, customCaption || undefined, customTags || undefined),
    onSuccess: (data) => {
      setUploadResult(data)
      toast.success('Image uploaded and analyzed successfully')
      setSelectedFile(null)
      setCustomCaption('')
      setCustomTags('')
    },
    onError: () => {
      toast.error('Failed to upload image')
    },
  })

  const uploadVideoMutation = useMutation({
    mutationFn: (file: File) => api.uploadVideo(file, customSummary || undefined),
    onSuccess: (data) => {
      setUploadResult(data)
      toast.success('Video uploaded and analyzed successfully')
      setSelectedFile(null)
      setCustomSummary('')
    },
    onError: () => {
      toast.error('Failed to upload video')
    },
  })

  const uploadDocumentMutation = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file, customSummary || undefined, customTags || undefined),
    onSuccess: (data) => {
      setUploadResult(data)
      toast.success('Document uploaded and analyzed successfully')
      setSelectedFile(null)
      setCustomSummary('')
      setCustomTags('')
    },
    onError: () => {
      toast.error('Failed to upload document')
    },
  })

  const isUploading = uploadImageMutation.isPending || uploadVideoMutation.isPending || uploadDocumentMutation.isPending

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      setSelectedFile(file)
      setUploadResult(null)
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setUploadResult(null)
    }
  }

  const handleUpload = () => {
    if (!selectedFile) return

    switch (activeTab) {
      case 'image':
        uploadImageMutation.mutate(selectedFile)
        break
      case 'video':
        uploadVideoMutation.mutate(selectedFile)
        break
      case 'document':
        uploadDocumentMutation.mutate(selectedFile)
        break
    }
  }

  const getAcceptTypes = () => {
    switch (activeTab) {
      case 'image':
        return 'image/jpeg,image/png,image/gif,image/webp'
      case 'video':
        return 'video/mp4,video/avi,video/mov,video/webm'
      case 'document':
        return '.pdf,.doc,.docx'
    }
  }

  const tabConfig = {
    image: {
      icon: <Image className="w-5 h-5" />,
      label: 'Images',
      description: 'Upload images for AI analysis and semantic search',
      accepts: 'JPEG, PNG, GIF, WebP'
    },
    video: {
      icon: <Video className="w-5 h-5" />,
      label: 'Videos',
      description: 'Upload videos for frame-by-frame analysis',
      accepts: 'MP4, AVI, MOV, WebM'
    },
    document: {
      icon: <FileText className="w-5 h-5" />,
      label: 'Documents',
      description: 'Upload documents for text extraction and summarization',
      accepts: 'PDF, DOC, DOCX'
    }
  }

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="section-header">
        <h2 className="section-title">Upload Content</h2>
        <p className="section-description">
          Upload images, videos, or documents for AI analysis and semantic indexing.
        </p>
      </div>

      {/* Upload Tabs */}
      <div className="upload-tabs">
        {(Object.keys(tabConfig) as UploadTab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => {
              setActiveTab(tab)
              setSelectedFile(null)
              setUploadResult(null)
            }}
            className={`upload-tab ${activeTab === tab ? 'active' : ''}`}
          >
            <span className="flex items-center justify-center gap-2">
              {tabConfig[tab].icon}
              {tabConfig[tab].label}
            </span>
          </button>
        ))}
      </div>

      {/* Upload Zone */}
      <div
        className={`dropzone ${isDragging ? 'active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept={getAcceptTypes()}
          onChange={handleFileSelect}
          className="hidden"
        />

        {selectedFile ? (
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-ddn-red-light flex items-center justify-center">
              {tabConfig[activeTab].icon}
            </div>
            <p className="font-medium text-neutral-900 mb-1">{selectedFile.name}</p>
            <p className="text-sm text-neutral-500">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
            <button
              onClick={(e) => { e.stopPropagation(); setSelectedFile(null) }}
              className="mt-4 text-sm text-ddn-red hover:underline flex items-center gap-1 mx-auto"
            >
              <X className="w-4 h-4" />
              Remove file
            </button>
          </div>
        ) : (
          <>
            <Upload className="dropzone-icon" />
            <p className="dropzone-text">
              Drag and drop your {activeTab} here, or click to browse
            </p>
            <p className="dropzone-hint">
              Accepts: {tabConfig[activeTab].accepts}
            </p>
          </>
        )}
      </div>

      {/* Custom Metadata */}
      {selectedFile && (
        <div className="card p-6 space-y-4">
          <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>
            Optional Metadata
          </h3>

          {/* Custom Fields */}
          {activeTab === 'image' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Custom Caption (Optional)
                </label>
                <input
                  type="text"
                  value={customCaption}
                  onChange={(e) => setCustomCaption(e.target.value)}
                  placeholder="Provide a custom caption for this image..."
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Custom Tags (Optional)
                </label>
                <input
                  type="text"
                  value={customTags}
                  onChange={(e) => setCustomTags(e.target.value)}
                  placeholder="e.g., outdoor, landscape, sunset"
                  className="input-field"
                />
                <p className="mt-1 text-xs text-neutral-500">
                  Separate tags with commas
                </p>
              </div>
            </div>
          )}

          {activeTab === 'video' && (
            <>
              <div>
                <label className="block text-xs font-medium text-neutral-500 uppercase tracking-wide mb-2">
                  Custom Summary
                </label>
                <textarea
                  value={customSummary}
                  onChange={(e) => setCustomSummary(e.target.value)}
                  placeholder="Override AI-generated summary (optional)"
                  className="input-field"
                  rows={3}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-neutral-500 uppercase tracking-wide mb-2">
                  Custom Tags
                </label>
                <input
                  type="text"
                  value={customTags}
                  onChange={(e) => setCustomTags(e.target.value)}
                  placeholder="tag1, tag2, tag3 (optional)"
                  className="input-field"
                />
              </div>
            </>
          )}

          {activeTab === 'document' && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-neutral-500 uppercase tracking-wide mb-2">
                  Custom Description
                </label>
                <textarea
                  value={customSummary}
                  onChange={(e) => setCustomSummary(e.target.value)}
                  placeholder="Override AI-generated summary (optional)"
                  className="input-field"
                  rows={3}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-neutral-500 uppercase tracking-wide mb-2">
                  Custom Tags
                </label>
                <input
                  type="text"
                  value={customTags}
                  onChange={(e) => setCustomTags(e.target.value)}
                  placeholder="tag1, tag2, tag3 (optional)"
                  className="input-field"
                />
              </div>
            </div>
          )}

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="btn-primary w-full"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing and Uploading...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Upload {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
              </>
            )}
          </button>
        </div>
      )}

      {/* Upload Result */}
      {uploadResult && (
        <div className="space-y-4">
          {/* Success Banner */}
          <div className="card p-6 bg-gradient-to-r from-nvidia-green/10 to-nvidia-green/5 border-l-4 border-nvidia-green">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-full bg-nvidia-green flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-nvidia-green mb-1">Upload Successful!</h3>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {uploadResult.message || 'Your file has been uploaded and analyzed successfully'}
                </p>
              </div>
            </div>
          </div>

          {/* Metadata Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Object Key */}
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-ddn-red" />
                <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">File Location</span>
              </div>
              <p className="text-sm font-mono text-neutral-700 dark:text-neutral-300 break-all">
                {uploadResult.object_key}
              </p>
            </div>

            {/* Duration (for videos) */}
            {'duration_seconds' in uploadResult && (
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Video className="w-4 h-4 text-ddn-red" />
                  <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">Duration</span>
                </div>
                <p className="text-2xl font-bold text-neutral-900 dark:text-white">
                  {uploadResult.duration_seconds?.toFixed(1)}s
                </p>
              </div>
            )}

            {/* Frame Count (for videos) */}
            {'frame_count' in uploadResult && (
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Image className="w-4 h-4 text-ddn-red" />
                  <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">Total Frames</span>
                </div>
                <p className="text-2xl font-bold text-neutral-900 dark:text-white">
                  {uploadResult.frame_count}
                </p>
              </div>
            )}

            {/* Detected Objects */}
            {'detected_objects' in uploadResult && uploadResult.detected_objects && (
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-4 h-4 text-nvidia-green" />
                  <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">Detected Objects</span>
                </div>
                <p className="text-sm text-neutral-700 dark:text-neutral-300">
                  {uploadResult.detected_objects}
                </p>
              </div>
            )}
          </div>

          {/* Summary Card */}
          {'summary' in uploadResult && uploadResult.summary && (
            <div className="card p-6">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="w-5 h-5 text-ddn-red" />
                <h4 className="font-semibold text-neutral-900 dark:text-white">AI Summary</h4>
              </div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 leading-relaxed">
                {uploadResult.summary}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Video Preview */}
      {uploadResult && 'duration_seconds' in uploadResult && uploadResult.object_key && (
        <div className="card p-6">
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Video Preview
          </h3>
          <video
            controls
            className="w-full rounded-lg bg-black"
            style={{ maxHeight: '500px' }}
            src={`/api/browse/video-stream/${uploadResult.object_key}`}
          >
            Your browser does not support the video tag.
          </video>
          <div className="mt-4 flex items-center gap-4 text-sm text-neutral-600">
            <span>Duration: {uploadResult.duration_seconds?.toFixed(1)}s</span>
            <span>•</span>
            <span>Frames: {'frame_count' in uploadResult ? uploadResult.frame_count : 'N/A'}</span>
            {'detected_objects' in uploadResult && uploadResult.detected_objects && (
              <>
                <span>•</span>
                <span>Detected: {uploadResult.detected_objects}</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="card p-6">
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          {tabConfig[activeTab].label} Upload Tips
        </h3>
        <ul className="space-y-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
          {activeTab === 'image' && (
            <>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
                Images are analyzed using CLIP for semantic embeddings
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
                BLIP generates automatic captions for each image
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
                Object detection identifies key elements in the image
              </li>
            </>
          )}
          {activeTab === 'video' && (
            <>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-nvidia-green rounded-full mt-2 flex-shrink-0" />
                Videos are analyzed frame-by-frame for comprehensive understanding
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-nvidia-green rounded-full mt-2 flex-shrink-0" />
                Key frames are extracted and indexed for search
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-nvidia-green rounded-full mt-2 flex-shrink-0" />
                Large videos may take longer to process
              </li>
            </>
          )}
          {activeTab === 'document' && (
            <>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-status-info rounded-full mt-2 flex-shrink-0" />
                Text is extracted from PDFs and Word documents
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-status-info rounded-full mt-2 flex-shrink-0" />
                AI summarization generates key insights automatically
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 bg-status-info rounded-full mt-2 flex-shrink-0" />
                Key terms are identified for improved search
              </li>
            </>
          )}
        </ul>
      </div>
    </div>
  )
}
