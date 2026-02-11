import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Search, Video, Loader2, Play, Clock, ChevronDown } from 'lucide-react'
import toast from 'react-hot-toast'

interface MatchingFrame {
    frame_index: number
    timestamp: number
    caption: string
    clip_score: number
    frame_base64: string
}

interface VideoFrameSearchResult {
    success: boolean
    video_key: string
    query: string
    matching_frames: MatchingFrame[]
    total_frames_analyzed: number
}

interface VideoInfo {
    key: string
    modality: string
    size_bytes: number
    last_modified: string
    metadata: {
        [key: string]: any
    }
    presigned_url?: string
}

export default function VideoSearchPage() {
    const [videoKey, setVideoKey] = useState('')
    const [query, setQuery] = useState('')
    const [threshold, setThreshold] = useState(0.20)
    const [searchResults, setSearchResults] = useState<VideoFrameSearchResult | null>(null)
    const [dropdownOpen, setDropdownOpen] = useState(false)

    // Fetch available videos
    const { data: videosData, isLoading: videosLoading } = useQuery({
        queryKey: ['videos'],
        queryFn: async () => {
            const response = await fetch('/api/browse/videos')
            if (!response.ok) throw new Error('Failed to fetch videos')
            return response.json()
        },
    })

    const videos = videosData?.objects || []

    const searchMutation = useMutation({
        mutationFn: async () => {
            const response = await fetch('/api/upload/video/search-frames', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_key: videoKey, query, threshold })
            })
            if (!response.ok) throw new Error('Search failed')
            return response.json()
        },
        onSuccess: (data: VideoFrameSearchResult) => {
            setSearchResults(data)
            if (data.matching_frames.length === 0) {
                toast('No matching frames found', { icon: '🔍' })
            } else {
                toast.success(`Found ${data.matching_frames.length} matching frames`)
            }
        },
        onError: () => {
            toast.error('Video frame search failed')
        },
    })

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault()
        if (!videoKey.trim() || !query.trim()) {
            toast.error('Please select a video and enter a search query')
            return
        }
        searchMutation.mutate()
    }

    const getThresholdLabel = () => {
        if (threshold >= 0.40) return "High precision"
        if (threshold >= 0.25) return "Medium precision"
        return "Low precision (more results)"
    }

    const getVideoDisplayName = (key: string) => {
        const parts = key.split('/')
        return parts[parts.length - 1] // Get filename only
    }

    return (
        <div className="space-y-8">
            {/* Section Header */}
            <div className="section-header">
                <h2 className="section-title">Video Frame Search</h2>
                <p className="section-description">
                    Search within video content using CLIP semantic similarity. Find specific frames matching your query.
                </p>
            </div>

            {/* Search Form */}
            <form onSubmit={handleSearch} className="card p-6 space-y-4">
                <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Select Video
                    </label>
                    <div className="relative">
                        <button
                            type="button"
                            onClick={() => setDropdownOpen(!dropdownOpen)}
                            className="input-field w-full text-left flex items-center justify-between"
                            disabled={videosLoading}
                        >
                            <span className={videoKey ? '' : 'text-neutral-400'}>
                                {videoKey ? getVideoDisplayName(videoKey) : 'Select a video...'}
                            </span>
                            <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                        </button>


                        {dropdownOpen && (
                            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                                {videosLoading ? (
                                    <div className="p-4 text-center text-neutral-500">
                                        <Loader2 className="w-5 h-5 animate-spin mx-auto" />
                                    </div>
                                ) : videos.length === 0 ? (
                                    <div className="p-4 text-center text-neutral-500 text-sm">
                                        No videos found. Upload a video first.
                                    </div>
                                ) : (
                                    videos.map((video: VideoInfo) => (
                                        <button
                                            key={video.key}
                                            type="button"
                                            onClick={() => {
                                                setVideoKey(video.key)
                                                setDropdownOpen(false)
                                            }}
                                            className="w-full px-4 py-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors text-sm"
                                        >
                                            <div className="font-medium">{getVideoDisplayName(video.key)}</div>
                                            <div className="text-xs text-neutral-500">
                                                {(video.size_bytes / 1024 / 1024).toFixed(1)} MB
                                            </div>
                                        </button>
                                    ))
                                )}
                            </div>
                        )}
                    </div>
                    <p className="mt-1 text-xs text-neutral-500">
                        {videos.length} video(s) available
                    </p>
                </div>

                <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Search Query
                    </label>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="person walking, car on road, sunset scene..."
                        className="input-field"
                    />
                </div>

                {/* Threshold Slider */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                            Match Threshold
                        </label>
                        <span className="text-sm font-semibold text-ddn-red">
                            {threshold.toFixed(2)} - {getThresholdLabel()}
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0.10"
                        max="0.70"
                        step="0.05"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer dark:bg-neutral-700"
                    />
                    <p className="mt-1 text-xs text-neutral-500">
                        Lower: more results, less precise. Higher: fewer results, more precise.
                    </p>
                </div>

                <button
                    type="submit"
                    disabled={searchMutation.isPending}
                    className="btn-primary w-full"
                >
                    {searchMutation.isPending ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Analyzing Frames...
                        </>
                    ) : (
                        <>
                            <Search className="w-4 h-4" />
                            Search Video Frames
                        </>
                    )}
                </button>
            </form>

            {/* Video Preview */}
            {videoKey && !searchResults && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold mb-4">Selected Video Preview</h3>
                    <video
                        controls
                        className="w-full rounded-lg bg-black"
                        style={{ maxHeight: '500px' }}
                        src={`/api/browse/video-stream/${videoKey}`}
                        preload="metadata"
                    >
                        Your browser does not support the video tag.
                    </video>
                    <p className="mt-3 text-sm text-neutral-500">
                        📁 {getVideoDisplayName(videoKey)}
                    </p>
                </div>
            )}

            {/* Search Results */}
            {searchResults && (
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold">
                            {searchResults.matching_frames.length} Matching Frames
                        </h3>
                        <span className="text-sm text-neutral-500">
                            Analyzed {searchResults.total_frames_analyzed} keyframes
                        </span>
                    </div>

                    {searchResults.matching_frames.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {searchResults.matching_frames.map((frame, idx) => (
                                <div key={idx} className="card p-4 space-y-3">
                                    {/* Frame Image */}
                                    <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                                        <img
                                            src={`data:image/jpeg;base64,${frame.frame_base64}`}
                                            alt={frame.caption}
                                            className="w-full h-full object-contain"
                                        />
                                        <div className="absolute top-2 right-2 flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-nvidia-green text-white">
                                            {(frame.clip_score * 100).toFixed(0)}%
                                        </div>
                                    </div>

                                    {/* Frame Details */}
                                    <div className="space-y-2">
                                        <div className="flex items-center gap-2 text-sm text-neutral-600 dark:text-neutral-400">
                                            <Clock className="w-4 h-4" />
                                            <span>{frame.timestamp.toFixed(1)}s</span>
                                            <span className="text-neutral-400">•</span>
                                            <Play className="w-4 h-4" />
                                            <span>Frame {frame.frame_index}</span>
                                        </div>
                                        <p className="text-sm text-neutral-700 dark:text-neutral-300">
                                            {frame.caption}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="card p-12 text-center">
                            <Video className="w-12 h-12 mx-auto mb-4 text-neutral-400" />
                            <p className="text-neutral-600 dark:text-neutral-400">
                                No frames matched your query with the current threshold.
                            </p>
                            <p className="text-sm text-neutral-500 mt-2">
                                Try lowering the threshold or using a different search query.
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
