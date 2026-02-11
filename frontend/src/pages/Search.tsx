import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Search as SearchIcon, Image, Video, FileText, Loader2, Download, Copy, FolderInput, X } from 'lucide-react'
import toast from 'react-hot-toast'

type Modality = 'all' | 'image' | 'video' | 'document'

interface StorageInfo {
  source: string
  storage_class: string
  access_control: {
    read: boolean
    write: boolean
    delete: boolean
  }
  protocol: string
  encryption?: string
  versioning_enabled: boolean
  etag?: string
  retrieval_time_ms?: number
}

interface SearchResult {
  object_key: string
  modality: string
  relevance_score: number
  size_bytes: number
  last_modified: string
  metadata: {
    [key: string]: unknown
  }
  presigned_url?: string
  storage_info?: StorageInfo
}

interface SearchResponse {
  success: boolean
  query: string
  results: SearchResult[]
  total_results: number
  search_time_ms?: number // Added this back as it was in the original success toast
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [modality, setModality] = useState<Modality>('video')
  const [threshold, setThreshold] = useState(0.30)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  // Removed searchTime state as per instruction, but keeping it for the toast message logic.
  // If the new API doesn't return search_time_ms, the toast should be adjusted.
  // For now, I'll keep the searchTime state and update the mutationFn to return it if possible,
  // or remove it from the toast if the new API doesn't provide it.
  // Based on the provided `Code Edit` for `searchMutation`, `search_time_ms` is not returned.
  // So, I will remove `searchTime` state and adjust the toast.

  // Modal state
  const [showCopyModal, setShowCopyModal] = useState(false)
  const [showMoveModal, setShowMoveModal] = useState(false)
  const [selectedObjectKey, setSelectedObjectKey] = useState('')
  const [destinationPath, setDestinationPath] = useState('')

  const searchMutation = useMutation({
    mutationFn: async ({ query, modality, threshold }: { query: string; modality: Modality; threshold: number }) => {
      const response = await fetch('/api/search/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, modality, threshold, limit: 20 })
      })
      if (!response.ok) throw new Error('Search failed')
      return response.json() as Promise<SearchResponse>
    },
    onSuccess: (data: SearchResponse) => {
      setSearchResults(data.results)
      // setSearchTime(data.search_time_ms) // Removed as per instruction and new mutationFn
      if (data.results.length === 0) {
        toast('No results found', { icon: '🔍' })
      } else {
        // Adjusted toast message since search_time_ms might not be available from the new API
        if (data.search_time_ms) {
          toast.success(`Found ${data.results.length} results in ${data.search_time_ms.toFixed(0)}ms`)
        } else {
          toast.success(`Found ${data.results.length} results`)
        }
      }
    },
    onError: () => {
      toast.error('Search failed')
    },
  })

  const copyMutation = useMutation({
    mutationFn: async ({ source, destination }: { source: string; destination: string }) => {
      const response = await fetch(`/api/browse/copy?source_key=${encodeURIComponent(source)}&destination_key=${encodeURIComponent(destination)}`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Copy failed')
      return response.json()
    },
    onSuccess: () => {
      toast.success('Object copied successfully')
      setShowCopyModal(false)
      setDestinationPath('')
    },
    onError: () => {
      toast.error('Failed to copy object')
    },
  })

  const moveMutation = useMutation({
    mutationFn: async ({ source, destination }: { source: string; destination: string }) => {
      const response = await fetch(`/api/browse/move?source_key=${encodeURIComponent(source)}&destination_key=${encodeURIComponent(destination)}`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Move failed')
      return response.json()
    },
    onSuccess: () => {
      toast.success('Object moved successfully')
      setShowMoveModal(false)
      setDestinationPath('')
      if (query) searchMutation.mutate({ query, modality, threshold })
    },
    onError: () => {
      toast.error('Failed to move object')
    },
  })

  const handleCopyClick = (objectKey: string) => {
    setSelectedObjectKey(objectKey)
    setDestinationPath(objectKey)
    setShowCopyModal(true)
  }

  const handleMoveClick = (objectKey: string) => {
    setSelectedObjectKey(objectKey)
    setDestinationPath(objectKey)
    setShowMoveModal(true)
  }

  const handleCopySubmit = () => {
    if (!destinationPath.trim()) return
    copyMutation.mutate({ source: selectedObjectKey, destination: destinationPath })
  }

  const handleMoveSubmit = () => {
    if (!destinationPath.trim()) return
    moveMutation.mutate({ source: selectedObjectKey, destination: destinationPath })
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    searchMutation.mutate({ query, modality, threshold })
  }

  const modalityOptions = [
    { value: 'video', label: 'Videos', icon: <Video className="w-4 h-4" /> },
    { value: 'image', label: 'Images', icon: <Image className="w-4 h-4" /> },
    { value: 'document', label: 'Documents', icon: <FileText className="w-4 h-4" /> },
  ]

  const getModalityIcon = (modalityType: Modality) => {
    const option = modalityOptions.find((opt) => opt.value === modalityType)
    return option?.icon || <SearchIcon className="w-12 h-12 text-neutral-400" />
  }

  const getThresholdLabel = () => {
    if (threshold >= 0.45) return "High precision (specific objects/people)"
    if (threshold >= 0.30) return "Medium precision (vehicles/scenes)"
    return "Low precision (broad categories)"
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="section-header">
        <h2 className="section-title">Semantic Search</h2>
        <p className="section-description">
          Search across your content using natural language. Find images, videos, and documents based on meaning.
        </p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="search-bar">
        <SearchIcon className="search-icon w-5 h-5" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe what you're looking for..."
          className="pr-28"
        />
        <button
          type="submit"
          disabled={searchMutation.isPending || !query.trim()}
          className="search-button"
        >
          {searchMutation.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            'Search'
          )}
        </button>
      </form>

      {/* Modality Filter and Threshold Slider */}
      <div className="space-y-4"> {/* New wrapper div for both */}
        <div className="flex items-center justify-center gap-2">
          {modalityOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setModality(option.value as Modality)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${modality === option.value
                ? 'bg-ddn-red text-white'
                : 'bg-surface-secondary text-secondary hover:bg-surface-tertiary'
                }`}
            >
              {option.icon}
              {option.label}
            </button>
          ))}
        </div>

        {/* Semantic Threshold Slider */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
              Semantic Match Threshold
            </label>
            <span className="text-sm font-semibold text-ddn-red">
              {threshold.toFixed(2)} - {getThresholdLabel()}
            </span>
          </div>
          <input
            type="range"
            min="0.20"
            max="0.70"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer dark:bg-neutral-700"
            style={{
              background: `linear-gradient(to right, var(--ddn-red) 0%,var(--ddn-red) ${((threshold - 0.20) / 0.50) * 100}%, rgb(229 229 229) ${((threshold - 0.20) / 0.50) * 100}%, rgb(229 229 229) 100%)`
            }}
          />
          <p className="mt-2 text-xs text-neutral-500">
            Lower values (0.20-0.30): Broader matches, more results. Higher values (0.45-0.70): Precise matches, fewer results.
          </p>
        </div>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 ? (
        <div className="media-grid">
          {searchResults.map((result) => (
            <div key={result.object_key} className="media-card">
              {/* Preview */}
              {result.modality === 'image' ? (
                <img
                  src={`/api/browse/video-stream/${result.object_key}`}
                  alt={String(result.metadata?.caption || 'Image')}
                  className="media-card-image"
                />
              ) : result.modality === 'video' ? (
                <video
                  controls
                  className="media-card-image"
                  src={`/api/browse/video-stream/${result.object_key}`}
                  preload="metadata"
                >
                  Your browser does not support the video tag.
                </video>
              ) : (
                <div className="media-card-image flex items-center justify-center">
                  {getModalityIcon(result.modality)}
                </div>
              )}

              {/* Content */}
              <div className="media-card-content">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="modality-badge">
                    {result.modality}
                  </span>
                  <span className="text-xs text-neutral-500">
                    Score: {(result.relevance_score * 100).toFixed(0)}%
                  </span>

                  {/* Storage Source Badge */}
                  {result.storage_info && (
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-medium"
                      style={{
                        backgroundColor: result.storage_info.source === 'local_cache'
                          ? '#fef3c7'
                          : result.storage_info.source === 'ddn_infinia'
                            ? '#dcfce7'
                            : '#dbeafe',
                        color: result.storage_info.source === 'local_cache'
                          ? '#92400e'
                          : result.storage_info.source === 'ddn_infinia'
                            ? '#166534'
                            : '#1e40af'
                      }}
                    >
                      📦 {result.storage_info.source === 'local_cache' ? 'Local Cache' :
                        result.storage_info.source === 'ddn_infinia' ? 'DDN INFINIA' : 'AWS S3'}
                    </span>
                  )}
                </div>

                <h3 className="text-sm font-semibold mb-2 line-clamp-1">
                  {result.object_key.split('/').pop()}
                </h3>

                {/* Storage Capabilities */}
                {result.storage_info && (
                  <div className="mb-3 p-2 bg-neutral-50 dark:bg-neutral-800/50 rounded-lg border border-neutral-200 dark:border-neutral-700">
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {/* Access Permissions */}
                      <div>
                        <span className="font-medium text-neutral-700 dark:text-neutral-300">Access:</span>
                        <div className="flex gap-1 mt-1">
                          {result.storage_info.access_control.read && (
                            <span className="px-1.5 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded text-[10px] font-medium">
                              ✓ Read
                            </span>
                          )}
                          {result.storage_info.access_control.write && (
                            <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded text-[10px] font-medium">
                              ✓ Write
                            </span>
                          )}
                          {result.storage_info.access_control.delete && (
                            <span className="px-1.5 py-0.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded text-[10px] font-medium">
                              ✓ Delete
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Storage Class */}
                      <div>
                        <span className="font-medium text-neutral-700 dark:text-neutral-300">Class:</span>
                        <div className="mt-1 text-[10px] text-neutral-600 dark:text-neutral-400 font-mono">
                          {result.storage_info.storage_class}
                        </div>
                      </div>

                      {/* Protocol & Encryption */}
                      <div>
                        <span className="font-medium text-neutral-700 dark:text-neutral-300">Protocol:</span>
                        <div className="mt-1 text-[10px] text-neutral-600 dark:text-neutral-400">
                          {result.storage_info.protocol}
                          {result.storage_info.encryption && (
                            <span className="ml-1 text-green-600 dark:text-green-400">🔒 {result.storage_info.encryption}</span>
                          )}
                        </div>
                      </div>

                      {/* Retrieval Time */}
                      {result.storage_info.retrieval_time_ms !== undefined && (
                        <div>
                          <span className="font-medium text-neutral-700 dark:text-neutral-300">Retrieval:</span>
                          <div className="mt-1 text-[10px] font-mono"
                            style={{
                              color: result.storage_info.retrieval_time_ms < 5 ? '#16a34a' :
                                result.storage_info.retrieval_time_ms < 10 ? '#2563eb' : '#dc2626'
                            }}
                          >
                            ⚡ {result.storage_info.retrieval_time_ms.toFixed(1)}ms
                          </div>
                        </div>
                      )}

                      {/* Versioning */}
                      {result.storage_info.versioning_enabled && (
                        <div className="col-span-2">
                          <span className="px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded text-[10px] font-medium">
                            📚 Versioning Enabled
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Caption/Summary */}
                {result.metadata?.caption && (
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-2 line-clamp-2">
                    {String(result.metadata.caption)}
                  </p>
                )}

                {/* Key Details - No JSON */}
                <div className="text-xs text-neutral-500 space-y-1">
                  {/* Detected Objects/Tags as badges */}
                  {result.metadata?.detected_objects && (
                    <div className="mb-2">
                      <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300 block mb-1.5">Detected Tags:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {String(result.metadata.detected_objects).split(',').map((tag, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-green-100 text-green-700 border border-green-200 rounded-md text-xs font-medium dark:bg-green-900/30 dark:text-green-400 dark:border-green-800"
                          >
                            🏷️ {tag.trim()}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* AI Summary */}
                  {result.metadata?.video_summary && (
                    <div className="mb-2">
                      <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300 block mb-1.5">AI Summary:</span>
                      <p className="text-xs text-neutral-600 dark:text-neutral-400 leading-relaxed">
                        {String(result.metadata.video_summary)}
                      </p>
                    </div>
                  )}

                  {result.modality === 'video' && result.metadata?.duration_seconds && (
                    <div>
                      <span className="font-medium">Duration:</span> {Number(result.metadata.duration_seconds).toFixed(1)}s
                    </div>
                  )}
                  {result.modality === 'document' && result.metadata?.word_count && (
                    <div>
                      <span className="font-medium">Words:</span> {String(result.metadata.word_count)}
                    </div>
                  )}
                  <div>
                    <span className="font-medium">Size:</span> {formatFileSize(result.size_bytes)}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 px-4 pb-4 border-t border-neutral-200 dark:border-neutral-700 pt-3">
                <a
                  href={`/api/browse/video-stream/${result.object_key}?download=true`}
                  download
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 rounded-lg transition-colors"
                  title="Download to local machine"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </a>

                <button
                  onClick={() => handleCopyClick(result.object_key)}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 rounded-lg transition-colors"
                  title="Copy to new location in S3"
                >
                  <Copy className="w-3.5 h-3.5" />
                  Copy
                </button>

                <button
                  onClick={() => handleMoveClick(result.object_key)}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium bg-nvidia-green hover:bg-nvidia-green/90 text-white rounded-lg transition-colors"
                  title="Move to new location in S3"
                >
                  <FolderInput className="w-3.5 h-3.5" />
                  Move
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : searchMutation.isSuccess ? (
        <div className="text-center py-16">
          <SearchIcon className="w-12 h-12 mx-auto mb-4 text-muted opacity-50" />
          <h3 className="font-semibold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>
            No results found
          </h3>
          <p className="text-muted">
            Try adjusting your search query or upload more content
          </p>
        </div>
      ) : !searchMutation.isPending ? (
        <div className="text-center py-16">
          <SearchIcon className="w-12 h-12 mx-auto mb-4 text-muted opacity-50" />
          <h3 className="font-semibold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>
            Search your content
          </h3>
          <p className="text-muted">
            Enter a query above to find relevant images, videos, and documents
          </p>
        </div>
      ) : null}

      {/* Example Queries */}
      <div className="card p-6">
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Example Queries
        </h3>
        <div className="flex flex-wrap gap-2">
          {[
            'sunset over mountains',
            'person using laptop',
            'data visualization chart',
            'product presentation',
            'quarterly report summary',
            'team meeting video'
          ].map((example) => (
            <button
              key={example}
              onClick={() => {
                setQuery(example)
                setModality('all')
              }}
              className="px-3 py-1.5 rounded-full text-sm bg-surface-secondary hover:bg-surface-tertiary transition-colors"
              style={{ color: 'var(--text-secondary)' }}
            >
              "{example}"
            </button>
          ))}
        </div>
      </div>

      {/* Copy Modal */}
      {showCopyModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowCopyModal(false)}>
          <div className="card p-6 max-w-lg w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Copy Object</h3>
              <button onClick={() => setShowCopyModal(false)} className="text-neutral-500 hover:text-neutral-700">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
              Source: <code className="bg-neutral-100 dark:bg-neutral-800 px-2 py-1 rounded text-xs">{selectedObjectKey}</code>
            </p>
            <input
              type="text"
              value={destinationPath}
              onChange={(e) => setDestinationPath(e.target.value)}
              placeholder="Enter destination path..."
              className="input-field mb-4"
              autoFocus
            />
            <div className="flex gap-3">
              <button onClick={() => setShowCopyModal(false)} className="flex-1 btn-secondary">
                Cancel
              </button>
              <button onClick={handleCopySubmit} disabled={copyMutation.isPending} className="flex-1 btn-primary">
                {copyMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Copy'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Move Modal */}
      {showMoveModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowMoveModal(false)}>
          <div className="card p-6 max-w-lg w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Move Object</h3>
              <button onClick={() => setShowMoveModal(false)} className="text-neutral-500 hover:text-neutral-700">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-red-600 dark:text-red-400 mb-2">
              ⚠️ Warning: This will delete the original file!
            </p>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
              Source: <code className="bg-neutral-100 dark:bg-neutral-800 px-2 py-1 rounded text-xs">{selectedObjectKey}</code>
            </p>
            <input
              type="text"
              value={destinationPath}
              onChange={(e) => setDestinationPath(e.target.value)}
              placeholder="Enter destination path..."
              className="input-field mb-4"
              autoFocus
            />
            <div className="flex gap-3">
              <button onClick={() => setShowMoveModal(false)} className="flex-1 btn-secondary">
                Cancel
              </button>
              <button onClick={handleMoveSubmit} disabled={moveMutation.isPending} className="flex-1 bg-nvidia-green hover:bg-nvidia-green/90 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2">
                {moveMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Move'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
